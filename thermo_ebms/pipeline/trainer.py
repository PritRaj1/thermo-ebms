from clu import metric_writers
from clu import periodic_actions
import jax
from flax import nnx
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path
from ml_collections import ConfigDict
import h5py
import yaml

from .loaders import get_loaders
from ..models import mleEBM, thermoEBM
from .opt import coupled_opt
from .jit import gen, eval_step, train_step


def to_uint8(x: jax.Array) -> np.ndarray:
	x = np.asarray(x)
	x = (x + 1.0) * 127.5
	x = np.rint(np.clip(x, 0, 255))
	return x.astype(np.uint8)


class ebmTrainer:
	def __init__(
		self,
		config: ConfigDict,
		rngs: nnx.Rngs,
	):
		self.model = (
			thermoEBM(config, rngs)
			if config.thermo.num_temps > 1
			else mleEBM(config, rngs)
		)

		self.train_loader, self.test_loader, updates_per_epoch = get_loaders(
			config.training
		)
		self.num_epochs = config.training.epochs
		self.final_samples = config.unbiased_metrics.num_samples
		self.final_bsize = config.unbiased_metrics.batch_size_to_generate

		ckpt_every = config.logging.ckpt_every * updates_per_epoch
		self.eval_every = config.logging.eval_every * updates_per_epoch
		self.sample_every = config.logging.sample_every * updates_per_epoch
		self.num_samples = config.logging.num_samples

		self.tx = coupled_opt(self.model, config, updates_per_epoch)
		graph, ps, st = nnx.split(self.model, nnx.Param, ...)
		self.opt_st = self.tx.init(ps)

		logdir = config.logging.logdir
		self.logdir = Path(logdir)
		self.logdir.mkdir(parents=True, exist_ok=True)
		self.writer = metric_writers.create_default_writer(logdir=logdir)
		with open(self.logdir / "config_copy.yaml", "w") as f:
			yaml.safe_dump(config.to_dict(), f)

		self.progress = periodic_actions.ReportProgress(
			num_train_steps=updates_per_epoch * self.num_epochs, writer=self.writer
		)

		self.ckpt_manager = ocp.CheckpointManager(
			config.logging.ckpt_dir,
			options=ocp.CheckpointManagerOptions(
				max_to_keep=5,
				save_interval_steps=ckpt_every,
				create=True,
			),
		)

	def train_epoch(self, key: jax.Array):
		for batch in self.train_loader:
			self.model, self.opt_st, loss, key = train_step(
				self.tx, self.opt_st, self.model, batch["x"], key
			)
			self.writer.write_scalars(self.model.train_idx, {"batch_loss": loss})
			self.progress(self.model.train_idx)

		train_idx = self.model.train_idx
		if train_idx % self.eval_every == 0:
			loss = 0.0
			num_batches = 0
			for batch in self.test_loader:
				loss_val, key = eval_step(self.model, batch["x"], key)
				loss += loss_val
				num_batches += 1

			self.writer.write_scalars(
				self.model.train_idx, {"test_loss": loss / num_batches}
			)

		if train_idx % self.sample_every == 0:
			x, key = gen(self.model, self.num_samples, key)
			self.writer.write_images(train_idx, {"generated_batch": x})

		self.ckpt_manager.save(
			train_idx,
			args=ocp.args.StandardSave(
				{
					"model": self.model,
					"opt_state": self.opt_st,
					"rng": key,
					"step": train_idx,
				}
			),
		)

		return key

	def run(self, key: jax.Array) -> jax.Array:
		for epoch in range(self.num_epochs):
			key = self.train_epoch(key)

		self.writer.flush()

		with h5py.File(self.logdir / "generated_samples.h5", "w") as f:
			x, key = gen(self.model, self.final_bsize, key)

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
				x, key = gen(self.model, bs, key)
				dataset[idx : idx + bs] = to_uint8(x)
				idx += bs

		self.ckpt_manager.wait_until_finished()
		return key
