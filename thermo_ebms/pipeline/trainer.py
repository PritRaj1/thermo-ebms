import os
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import yaml
from clu import metric_writers, periodic_actions
from flax import nnx
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from ..config import Config
from ..models import mleEBM, mleKAEM, thermoEBM, thermoKAEM
from .loaders import get_loaders
from .opt import coupled_opt

cmap = plt.get_cmap("plasma")


def to_uint8(x: jax.Array) -> np.ndarray:
	x = jax.device_get(x)
	x = (x + 1.0) * 127.5
	x = np.rint(np.clip(x, 0, 255))
	return x.astype(np.uint8)


def loss_fn(
	m: nnx.Module, x: jax.Array, z_post: jax.Array, z_prior: jax.Array
) -> jax.Array:
	return m.loss(x, z_post, z_prior)


@nnx.jit
def update(
	state: nnx.ModelAndOptimizer,
	x: jax.Array,
	z_post: jax.Array,
	z_prior: jax.Array,
) -> tuple[jax.Array, jax.Array]:
	loss, grads = nnx.value_and_grad(loss_fn)(state.model, x, z_post, z_prior)
	state.update(grads)
	grad_norm = jnp.sqrt(
		sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
	)
	return loss, grad_norm


class ebmTrainer:
	def __init__(
		self,
		config: Config,
	):
		self.model_type = config.model.base.lower()
		model_cls = {
			("neural", True): thermoEBM,
			("neural", False): mleEBM,
			("kaem", True): thermoKAEM,
			("kaem", False): mleKAEM,
		}[(self.model_type, config.model.thermo.num_temps > 1)]

		# Distributed data parallel sharding
		self.mesh = jax.make_mesh((jax.device_count(),), ("data",))
		nnx.use_eager_sharding(True)
		self.batch_sharding = NamedSharding(self.mesh, P("data", None, None, None))

		self.train_loader, self.updates_per_epoch = get_loaders(
			config.training, config.model.seed
		)

		key_init = jax.random.key(config.model.seed)
		self.num_epochs = config.training.epochs

		with jax.set_mesh(self.mesh):
			key = nnx.Rngs(key_init)
			model = model_cls(config.model, key)
			tx = coupled_opt(config.optim, self.updates_per_epoch * self.num_epochs)
			self.st = nnx.ModelAndOptimizer(model, tx, wrt=nnx.Param)

		self.final_samples = config.unbiased_metrics.num_samples
		self.final_bsize = config.unbiased_metrics.batch_size_to_generate
		ckpt_every = config.logging.ckpt_every * self.updates_per_epoch
		self.sample_every = config.logging.sample_every
		self.num_samples = config.logging.num_samples

		self.is_host0 = jax.process_index() == 0
		logdir = config.logging.logdir
		self.logdir = Path(logdir)
		if self.is_host0:
			self.logdir.mkdir(parents=True, exist_ok=True)
			self.writer = metric_writers.create_default_writer(logdir=logdir)
			with open(self.logdir / "config_copy.yaml", "w") as f:
				yaml.safe_dump(OmegaConf.to_container(config, resolve=True), f)
		else:
			self.writer = metric_writers.MultiWriter([])  # no-op

		self.progress = periodic_actions.ReportProgress(
			num_train_steps=self.updates_per_epoch * self.num_epochs, writer=self.writer
		)

		self.ckpt_manager = ocp.CheckpointManager(
			os.path.abspath(config.logging.ckpt_dir),
			options=ocp.CheckpointManagerOptions(
				max_to_keep=5,
				save_interval_steps=ckpt_every,
				create=True,
			),
		)

		self.profiler = periodic_actions.Profile(
			logdir=self.logdir,
		)

	def plot_kaem(self, step: int) -> None:
		model = self.st.model
		model.eval()
		ebm = model.ebm

		z_grid = ebm.nodes
		sigma = ebm.sigma
		log_p0 = (
			-0.5 * (z_grid / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
		)

		unnormalized_pdf = jnp.exp(ebm(z_grid) + log_p0)
		pdf = unnormalized_pdf / jnp.sum(
			unnormalized_pdf * ebm.weights, axis=0, keepdims=True
		)
		ref_pdf = (1.0 / (sigma * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
			-0.5 * (z_grid / sigma) ** 2
		)

		z_np = np.asarray(z_grid)
		pdf_np = np.asarray(pdf)
		ref_np = np.asarray(ref_pdf)

		images = []
		for q_idx in range(4):
			for p_idx in range(4):
				z_qp = z_np[:, 0, q_idx, p_idx]
				pdf_qp = pdf_np[:, 0, q_idx, p_idx]
				ref_qp = ref_np[:, 0, q_idx, p_idx]
				colour = cmap((q_idx * 4 + p_idx) / max(1, 16))

				fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
				ax.plot(
					z_qp,
					ref_qp,
					color="#7f7f7f",
					linestyle="--",
					linewidth=3.0,
					label=r"Ref $\mathcal{N}(0, 1)$",
				)
				ax.fill_between(
					z_qp,
					ref_qp,
					color="#7f7f7f",
					alpha=0.25,
					label="_nolegend_",
				)

				ax.plot(
					z_qp,
					pdf_qp,
					color=colour,
					linewidth=2.0,
					label="KAN density",
				)
				ax.fill_between(
					z_qp,
					pdf_qp,
					color=colour,
					alpha=0.35,
					label="_nolegend_",
				)

				ax.set_title(f"KAEM Density, (Q={q_idx},P={p_idx}) (Epoch {step})")
				ax.set_xlabel("z")
				ax.set_ylabel("PDF")
				ax.set_ylim(bottom=0.0, top=1.0)
				ax.grid(True, linestyle=":", alpha=0.6)
				ax.legend(loc="upper right", frameon=True)
				fig.tight_layout()

				fig.canvas.draw()
				images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
				plt.close(fig)

		self.writer.write_images(step, {"latents/densities": np.stack(images, axis=0)})

		if ebm.mixture:
			z_max = jnp.max(pdf, axis=0).reshape(ebm.Q, 1, 1, ebm.P)
			img = model.gen(z_max)
			self.writer.write_images(step, {"images/maxpdf_mixture": to_uint8(img)})

	def train_step(
		self, x: jax.Array, train_idx: int, key: jax.Array
	) -> tuple[jax.Array, jax.Array]:
		key, prior_key, posterior_key, grid_key = jax.random.split(key, 4)
		z_prior = self.st.model.sample_prior(prior_key, x.shape[0])
		z_post = self.st.model.sample_posterior(posterior_key, x)

		self.st.model.train()
		loss, grad_norm = update(self.st, x, z_post, z_prior)
		with jax.set_mesh(self.mesh):
			self.st.model.adapt_temps(
				train_idx, self.updates_per_epoch * self.num_epochs
			)
			self.st.model.adapt_domain(grid_key, z_post, train_idx)

		return loss, grad_norm, z_prior, z_post, key

	def train_epoch(self, key: jax.Array, epoch: int) -> jax.Array:
		train_idx = epoch * self.updates_per_epoch
		for i, batch in zip(range(self.updates_per_epoch), self.train_loader):
			x = jax.device_put(batch["x"], self.batch_sharding)
			key, subkey = jax.random.split(key)
			loss, grad_norm, z_prior, z_post, key = self.train_step(
				x, train_idx, subkey
			)
			self.profiler(train_idx)

			train_idx += 1

			if self.is_host0:
				self.writer.write_scalars(train_idx, {"batch_loss": loss})
				self.writer.write_scalars(train_idx, {"grad_norm": grad_norm})
				self.progress(train_idx)

		if (epoch % self.sample_every == 0) and self.is_host0:
			sample_key = jax.random.fold_in(key, train_idx)
			x, _ = self.st.model(sample_key, self.num_samples)
			self.writer.write_images(train_idx, {"images/generated": to_uint8(x)})
			self.writer.write_histograms(
				train_idx,
				{
					"latents/z_posterior": np.asarray(z_post),
					"latents/z_prior": np.asarray(z_prior),
				},
			)
			ps_histograms = {
				f"ebm_params/{'/'.join(map(str, path))}": np.asarray(val)
				for path, val in jax.tree_util.tree_leaves_with_path(
					nnx.state(self.st.model.ebm, nnx.Param)
				)
			}
			self.writer.write_histograms(train_idx, ps_histograms)

			if self.model_type == "kaem":
				self.plot_kaem(epoch)

		self.ckpt_manager.save(
			train_idx,
			args=ocp.args.StandardSave(
				{
					"train_state": nnx.state(self.st),
					"rng": key,
					"step": train_idx,
				}
			),
		)

		return key

	def run(self, key: jax.Array) -> jax.Array:
		for epoch in range(self.num_epochs):
			key = self.train_epoch(key, epoch)

		self.writer.flush()
		sync_global_devices("post_training_sync")

		final_step = self.num_epochs * self.updates_per_epoch
		if self.ckpt_manager.latest_step() != final_step:
			self.ckpt_manager.save(
				final_step,
				args=ocp.args.StandardSave(
					{
						"train_state": nnx.state(self.st),
						"rng": key,
						"step": final_step,
					}
				),
				force=True,
			)

		self.ckpt_manager.wait_until_finished()

		if self.is_host0 and (self.model_type == "kaem") and self.st.model.ebm.mixture:
			lut = self.st.model.make_lut()
			np.save(self.logdir / "inv_cdf_lut.npy", lut)
			alpha = np.asarray(self.st.model.ebm.alpha.squeeze())
			np.save(self.logdir / "mixture_alpha.npy", alpha)

		if self.is_host0:
			with h5py.File(self.logdir / "generated_samples.h5", "w") as f:
				sample_key = jax.random.fold_in(key, self.final_samples)
				x, _ = self.st.model(sample_key, self.final_bsize)

				dataset = f.create_dataset(
					"samples",
					shape=(self.final_samples, *x.shape[1:]),
					dtype=np.uint8,
				)
				f.attrs["num_samples"] = self.final_samples
				f.attrs["shape"] = x.shape
				f.attrs["dtype"] = "uint8"

				dataset[: len(x)] = to_uint8(x)
				idx = len(x)
				while idx < self.final_samples:
					bs = min(self.final_bsize, self.final_samples - idx)
					x, key = self.st.model(key, bs)
					dataset[idx : idx + bs] = to_uint8(x)
					idx += bs

		sync_global_devices("post_gen_sync")
		self.ckpt_manager.wait_until_finished()
		return key
