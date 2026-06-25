import h5py
import yaml
import jax
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
from .jit import eval_step, train_step
from ..config import Config


def to_uint8(x: jax.Array) -> np.ndarray:
	x = np.asarray(x)
	x = (x + 1.0) * 127.5
	x = np.rint(np.clip(x, 0, 255))
	return x.astype(np.uint8)


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
		self.train_loader, self.test_loader, self.updates_per_epoch = get_loaders(
			config.training, config.model.seed
		)

		key_init = jax.random.key(config.model.seed)

		with jax.set_mesh(self.mesh):
			key = nnx.Rngs(key_init)
			self.model = model_cls(config.model, key)
			graph, ps, st = nnx.split(self.model, nnx.Param, ...)
			self.tx = coupled_opt(
				self.model, config.model, config.lr_schedule, self.updates_per_epoch
			)
			self.opt_st = self.tx.init(ps)

		self.num_epochs = config.training.epochs
		self.final_samples = config.unbiased_metrics.num_samples
		self.final_bsize = (
			config.unbiased_metrics.batch_size_to_generate // jax.process_count()
		)

		ckpt_every = config.logging.ckpt_every * self.updates_per_epoch
		self.eval_every = config.logging.eval_every
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
			config.logging.ckpt_dir,
			options=ocp.CheckpointManagerOptions(
				max_to_keep=5,
				save_interval_steps=ckpt_every,
				create=True,
			),
		)

	def train_epoch(self, key: jax.Array, epoch: int):
		for _, batch in zip(range(self.updates_per_epoch), self.train_loader):
			x_sharded = jax.device_put(batch["x"], self.batch_sharding)
			key_idx = jax.random.fold_in(key, int(self.model.train_idx))
			self.model, self.opt_st, loss, _ = train_step(
				self.tx, self.opt_st, self.model, x_sharded, key_idx
			)
			if self.is_host0:
				self.writer.write_scalars(
					self.model.train_idx,
					{"batch_loss": loss},
				)
				self.progress(self.model.train_idx)

		if epoch % self.eval_every == 0:
			running_loss = 0.0
			num_batches = 0
			for batch in self.test_loader:
				x_sharded = jax.device_put(batch["x"], self.batch_sharding)
				eval_key = jax.random.fold_in(key_idx, num_batches)
				loss, _ = eval_step(self.model, x_sharded, eval_key)
				running_loss += loss
				num_batches += 1

			if self.is_host0:
				self.writer.write_scalars(
					self.model.train_idx, {"test_loss": running_loss / num_batches}
				)

		if (epoch % self.sample_every == 0) and self.is_host0:
			x, key = self.model(key, self.num_samples)
			self.writer.write_images(self.model.train_idx, {"generated_batch": x})
			del x

		if self.is_host0:
			self.ckpt_manager.save(
				self.model.train_idx,
				args=ocp.args.StandardSave(
					{
						"model": nnx.state(self.model),
						"opt_state": self.opt_st,
						"rng": key,
						"step": self.model.train_idx,
					}
				),
			)

		return jax.random.split(key, 2)[0]  # increment key

	def run(self, key: jax.Array) -> jax.Array:
		for epoch in range(self.num_epochs):
			key = self.train_epoch(key, epoch)

		self.writer.flush()
		sync_global_devices("post_training_sync")

		if self.is_host0:
			with h5py.File(self.logdir / "generated_samples.h5", "w") as f:
				x, key = self.model(key, self.final_bsize)

				dataset = f.create_dataset(
					"samples",
					shape=(self.final_samples, *x.shape[1:]),
					dtype=np.uint8,
					compression="gzip",
					compression_opts=4,
				)
				f.attrs["num_samples"] = self.final_samples
				f.attrs["num_updates"] = int(self.model.train_idx)
				f.attrs["shape"] = x.shape
				f.attrs["dtype"] = "uint8"

				dataset[: len(x)] = to_uint8(x)
				idx = len(x)
				while idx < self.final_samples:
					bs = min(self.final_bsize, self.final_samples - idx)
					x, key = self.model(key, bs)
					dataset[idx : idx + bs] = to_uint8(x)
					idx += bs

		sync_global_devices("post_gengen_sync")
		self.ckpt_manager.wait_until_finished()
		return key
