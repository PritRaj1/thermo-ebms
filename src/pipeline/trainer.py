from clu import metric_writers
from clu import periodic_actions
from flax import nnx
import orbax.checkpoint as ocp
import h5py
import yaml
from pathlib import Path

from .loaders import get_loaders
from ..networks import mleEBM, thermoEBM
from .opt import coupled_opt
from .metrics import UnbiasedMetrics


def load_config(config_path: str) -> ml_collections.ConfigDict:
	"""Loads YAML configuration into a ConfigDict."""
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return ml_collections.ConfigDict(config)


def to_uint8(x: jax.Array) -> np.ndarray:
	x = np.asarray(x)
	x = (x + 1.0) * 127.5
	x = np.rint(np.clip(x, 0, 255))
	return x.astype(np.uint8)


class ebmTrainer:
	def __init__(
		self,
		config_path: str,
		rngs: nnx.Rngs,
	):
		config = load_config(config_path)
		model = (
			thermoEBM(config, rngs)
			if config.thermo.num_temps > 1
			else mleEBM(config, rngs)
		)

		opt = coupled_opt(config)
		self.st = nnx.ModelAndOptimizer(model, opt)
		nnx.display(self.st.model)

		logdir = config.logging.logdir
		self.logdir = Path(logdir)
		self.logdir.mkdir(parents=True, exist_ok=True)
		self.writer = metric_writers.create_default_writer(logdir=logdir)
		with open(self.logdir / "config_copy.yaml", "w") as f:
			yaml.safe_dump(config.to_dict(), f)

		self.progress = periodic_actions.ReportProgress(
			num_train_steps=model.num_updates, writer=self.writer
		)

		epoch_updates = config.training.numdata // config.training.batch_size
		ckpt_every = config.logging.ckpt_every * epoch_updates
		self.eval_every = config.logging.eval_every * epoch_updates
		self.sample_every = config.logging.sample_every * epoch_updates
		self.num_samples = config.logging.num_samples

		self.ckpt_manager = ocp.CheckpointManager(
			config.logging.ckpt_dir,
			options=ocp.CheckpointManagerOptions(
				max_to_keep=5,
				save_interval_steps=ckpt_every,
				create=True,
			),
		)
		self.train_loader, self.test_loader = get_loaders(config.dataset)
		self.num_epochs = config.training.num_epochs
		self.final_samples = config.unbiased_metrics.num_samples
		self.final_bsize = config.unbiased_metrics.batch_size_to_generate

	@nnx.jit
	def update(self, key: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
		key, prior_key, posterior_key = jax.random.split(key, 3)
		z_prior = self.st.model.sample_prior(prior_key, x.shape[0])
		z_posterior = self.st.model.sample_posterior(posterior_key, x)

		graph, ps, st = nnx.split(self.st.model, nnx.Param, ...)

		def loss(ps_mut):
			model = nnx.merge(graph, ps_mut, st)
			cd = model.ebm.loss(z_posterior, z_prior)
			recon = model.loss(x, z_posterior, z_prior)
			return cd + recon

		loss_val, grads = nnx.value_and_grad(loss)(ps)
		self.st.update(grads)
		self.st.model.train_idx += 1
		return loss_val, key

	@nnx.jit
	def eval_step(self, key: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		z_prior = self.st.model.sample_prior(subkey, x.shape[0])
		return self.st.model.gen.loss(x, z_prior), subkey

	def train_epoch(self, key: jax.Array):
		for batch in self.train_loader:
			loss, key = self.update(key, batch["x"])
			self.writer.write_scalars(self.st.model.train_idx, {"train_loss": loss})
			self.progress(self.st.model.train_idx)

		train_idx = self.st.model.train_idx
		if train_idx % self.eval_every == 0:
			loss = 0.0
			for batch in self.test_loader:
				loss_val, key = self.eval_step(key, batch["x"])
				loss += loss_val

			loss /= len(self.test_loader)
			self.writer.write_scalars(self.st.model.train_idx, {"test_loss": loss})

		if train_idx % self.sample_every == 0:
			key, subkey = jax.random.split(key)
			x = self.st.model(subkey, self.num_samples)
			self.writer.write_images(train_idx, {"generated_batch": x})

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
			key = self.train_epoch(key)

		self.writer.flush()

		with h5py.File(self.logdir / "generated_samples.h5", "w") as f:
			key, subkey = jax.random.split(key)
			x = to_uint8(self.st.model(subkey, self.final_bsize))

			dataset = f.create_dataset(
				"samples",
				shape=(self.final_samples, *x.shape[1:]),
				dtype=np.uint8,
				compression="gzip",
				compression_opts=4,
			)
			f.attrs["num_samples"] = self.final_samples
			f.attrs["num_updates"] = int(self.st.model.train_idx)
			f.attrs["shape"] = x.shape
			f.attrs["dtype"] = "uint8"

			dataset[: len(x)] = x
			idx = len(x)
			while idx < self.final_samples:
				bs = min(self.final_bsize, self.final_samples - idx)
				key, subkey = jax.random.split(key)
				x = to_uint8(self.st.model(subkey, bs))
				dataset[idx : idx + bs] = x
				idx += bs

		return key
