import h5py
import yaml
import jax
import os
from flax import nnx
import numpy as np
import orbax.checkpoint as ocp
from clu import metric_writers
from clu import periodic_actions
from pathlib import Path
from omegaconf import OmegaConf
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import sync_global_devices

from .loaders import get_loaders
from ..models import mleEBM, mleKAEM, thermoEBM, thermoKAEM
from .opt import coupled_opt
from ..config import Config


def to_uint8(x: jax.Array) -> np.ndarray:
	x = jax.device_get(x)
	x = (x + 1.0) * 127.5
	x = np.rint(np.clip(x, 0, 255))
	return x.astype(np.uint8)


@nnx.jit
def update(
	state: nnx.ModelAndOptimizer,
	x: jax.Array,
	z_post: jax.Array,
	z_prior: jax.Array,
) -> tuple[nnx.ModelAndOptimizer, jax.Array]:

	def loss_fn(model):
		return model.ebm.loss(z_post, z_prior) + model.loss(x, z_post, z_prior)

	loss, grads = nnx.value_and_grad(loss_fn)(state.model)
	state.update(grads)
	return state, loss


class ebmTrainer:
	def __init__(
		self,
		config: Config,
	):
		model_cls = {
			("neural", True): thermoEBM,
			("neural", False): mleEBM,
			("kaem", True): thermoKAEM,
			("kaem", False): mleKAEM,
		}[(config.model.base.lower(), config.model.thermo.num_temps > 1)]

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
			opt = coupled_opt(
				model, config.model, config.lr_schedule, self.updates_per_epoch
			)
			self.st = nnx.ModelAndOptimizer(model, opt)

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

	def train_step(
		self, x: jax.Array, train_idx: int, key: jax.Array
	) -> tuple[jax.Array, jax.Array]:
		key, prior_key, posterior_key = jax.random.split(key, 3)
		z_prior = self.st.model.sample_prior(prior_key, x.shape[0])
		z_post = self.st.model.sample_posterior(posterior_key, x)

		if self.st.model.num_temps > 1:
			self.st.model.adapt_temps(x, z_post)

		if (self.st.model.base == "kaem") and hasattr(
			self.st.model.ebm.f.layers[0], "grid"
		):
			self.st.model.update_grid(z_post, train_idx)

		self.st.model.train()
		self.st, loss = update(self.st, x, z_post, z_prior)
		return loss, key

	def train_epoch(self, key: jax.Array, epoch: int) -> jax.Array:
		train_idx = epoch * self.updates_per_epoch
		for i, batch in zip(range(self.updates_per_epoch), self.train_loader):
			x = jax.device_put(batch["x"], self.batch_sharding)
			key, subkey = jax.random.split(key)
			loss, key = self.train_step(x, train_idx, subkey)

			train_idx += 1
			loss_val = float(jax.device_get(loss))

			if self.is_host0:
				self.writer.write_scalars(train_idx, {"batch_loss": loss_val})
				self.progress(train_idx)

		if (epoch % self.sample_every == 0) and self.is_host0:
			x, key = self.st.model(key, self.num_samples)
			self.writer.write_images(train_idx, {"generated_batch": to_uint8(x)})

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
					compression="gzip",
					compression_opts=4,
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

		sync_global_devices("post_gengen_sync")
		self.ckpt_manager.wait_until_finished()
		return key
