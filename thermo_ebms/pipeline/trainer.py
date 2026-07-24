import h5py
import yaml
import jax
import os
import numpy as np
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
import orbax.checkpoint as ocp
from clu import metric_writers
from clu import periodic_actions
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import sync_global_devices

from .opt import coupled_opt
from .loaders import get_loaders
from ..models import mleEBM, mleKAEM, thermoEBM, thermoKAEM
from ..config import Config


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
	grads_flat = jnp.concatenate(
		[g.flatten() for g in jax.tree_util.tree_leaves(grads)]
	)
	return loss, jnp.linalg.norm(grads_flat)


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
		self.mesh = Mesh(jax.devices(), axis_names=("data",))
		nnx.use_eager_sharding(True)
		self.batch_sharding = NamedSharding(self.mesh, P("data", None, None, None))
		self.train_loader, self.updates_per_epoch = get_loaders(
			config.training, config.model.seed
		)

		key_init = jax.random.key(config.model.seed)

		with jax.set_mesh(self.mesh):
			key = nnx.Rngs(key_init)
			model = model_cls(config.model, key)
			tx = coupled_opt(config.optim, self.updates_per_epoch)
			self.st = nnx.ModelAndOptimizer(model, tx, wrt=nnx.Param)

		self.num_epochs = config.training.epochs
		self.final_samples = config.unbiased_metrics.num_samples
		self.final_bsize = (
			config.unbiased_metrics.batch_size_to_generate // jax.process_count()
		)

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

	def plot_kaem_component(self, step: int) -> np.ndarray:
		model = self.st.model
		model.eval()
		ebm = model.ebm

		z_grid = jnp.linspace(-3.0, 3.0, num=200)
		z = jnp.repeat(
			jnp.repeat(jnp.expand_dims(z_grid, axis=(1, 2, 3)), ebm.Q, axis=-2),
			ebm.P,
			axis=-1,
		)
		f = ebm(z)[:, 1, 1, 1]  # Q = 1, P = 1 component
		log_p0 = (
			-0.5 * (z_grid / ebm.sigma) ** 2
			- jnp.log(ebm.sigma)
			- 0.5 * jnp.log(2.0 * jnp.pi)
		)

		unnormalized_pdf = jnp.exp(f + log_p0)
		quad = ebm(
			jnp.repeat(ebm.nodes, ebm.Q, axis=-2),
		)
		Z = jnp.sum(
			ebm.weights * jnp.exp(quad + ebm.log_p0(ebm.nodes)),
			axis=0,
			keepdims=True,
		)[:, 1, 1, 1]
		pdf = unnormalized_pdf / Z
		ref_pdf = (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * z_grid**2)

		z_np = np.asarray(z_grid)
		pdf_np = np.asarray(pdf)
		ref_np = np.asarray(ref_pdf)

		fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
		ax.plot(
			z_np,
			ref_np,
			color="#7f7f7f",
			linestyle="--",
			linewidth=3.0,
			label=r"Ref $\mathcal{N}(0, 1)$",
		)
		ax.fill_between(z_np, ref_np, color="#7f7f7f", alpha=0.25, label="_nolegend_")

		ax.plot(
			z_np,
			pdf_np,
			color="#1f77b4",
			linewidth=2.0,
			label="KAEM Component)",
		)
		ax.fill_between(z_np, pdf_np, color="#1f77b4", alpha=0.35, label="_nolegend_")

		ax.set_title(f"Density, (Q=1,P=1) (Epoch {step})")
		ax.set_xlabel("z")
		ax.set_ylabel("PDF")
		ax.set_xlim([-3.0, 3.0])
		ax.set_ylim(bottom=0.0)
		ax.grid(True, linestyle=":", alpha=0.6)
		ax.legend(loc="upper right", frameon=True)
		fig.tight_layout()

		fig.canvas.draw()
		image_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
		plt.close(fig)
		return image_array

	def train_step(
		self, x: jax.Array, train_idx: int, key: jax.Array
	) -> tuple[jax.Array, jax.Array]:
		key, prior_key, posterior_key = jax.random.split(key, 3)
		z_prior = self.st.model.sample_prior(prior_key, x.shape[0])
		z_post = self.st.model.sample_posterior(posterior_key, x)

		self.st.model.train()
		loss, grad_norm = update(self.st, x, z_post, z_prior)
		self.st.model.adapt_temps(train_idx, self.updates_per_epoch * self.num_epochs)
		return loss, grad_norm, key

	def train_epoch(self, key: jax.Array, epoch: int) -> jax.Array:
		train_idx = epoch * self.updates_per_epoch
		for i, batch in zip(range(self.updates_per_epoch), self.train_loader):
			x = jax.device_put(batch["x"], self.batch_sharding)
			key, subkey = jax.random.split(key)
			loss, grad_norm, key = self.train_step(x, train_idx, subkey)
			self.profiler(train_idx)

			train_idx += 1

			if self.is_host0:
				self.writer.write_scalars(train_idx, {"batch_loss": loss})
				self.writer.write_scalars(train_idx, {"grad_norm": grad_norm})
				self.progress(train_idx)

		if (epoch % self.sample_every == 0) and self.is_host0:
			x, key = self.st.model(key, self.num_samples)
			self.writer.write_images(train_idx, {"generated_batch": to_uint8(x)})

			if self.model_type == "kaem":
				self.writer.write_images(
					train_idx, {"kaem_density": self.plot_kaem_component(epoch)}
				)

		if self.is_host0:
			self.ckpt_manager.save(
				train_idx,
				args=ocp.args.StandardSave(
					{
						"train_state": self.st,
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

		if self.is_host0:
			with h5py.File(self.logdir / "generated_samples.h5", "w") as f:
				x, key = self.st.model(key, self.final_bsize)

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
