from clu import metric_writers
from clu import periodic_actions

import orbax.checkpoint as ocp
from flax import nnx
from .loaders import get_loaders
from ..networks import mleEBM, thermoEBM
from .opt import coupled_opt


def load_config(config_path: str) -> ml_collections.ConfigDict:
	"""Loads YAML configuration into a ConfigDict."""
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return ml_collections.ConfigDict(config)


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

		self.writer = metric_writers.create_default_writer(logdir=config.logging.logdir)
		with open(logdir / "config_copy.yaml", "w") as f:
			yaml.safe_dump(self.config.to_dict(), f)

		self.progress = periodic_actions.ReportProgress(
			num_train_steps=model.num_updates
		)

		epoch_updates = config.training.numdata // config.training.batch_size
		self.ckpt_every = config.logging.ckpt_every * epoch_updates
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

	@nnx.jit
	def update(self, key: jax.Array, x: jax.Array) -> jax.Array:
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
		return loss_val, subkey

	@nnx.jit
	def eval_step(self, key: jax.Array, x: jax.Array) -> jax.Array:
		key, subkey = jax.random.split(key)
		z_prior = self.st.model.sample_prior(subkey, x.shape[0])
		return self.st.model.gen.loss(x, z_prior), subkey

	def train(self, key: jax.Array):
		for batch in self.train_loader:
			loss, key = self.update(key, batch["x"])
			self.writer.write_scalars(self.st.model.train_idx, {"train_loss": loss})

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

		if train_idx % self.ckpt_every == 0:
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
