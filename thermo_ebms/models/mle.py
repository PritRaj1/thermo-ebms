import jax
from flax import nnx

from ..config import ULAConfig
from .base import neuralEBM
from .kaem import KAEM
from .sampling import ula_sampler


class MLE:
	def posterior_score(self, z: jax.Array, minibatch: jax.Array) -> jax.Array:
		return self.gen.llhood_score(z, minibatch) + self.ebm.prior_score(z)

	def mcmc_setup(self, config: ULAConfig):
		self.posterior_sampler = ula_sampler(config)

	@nnx.jit
	def _sample_posterior(
		self, key: jax.Array, z: jax.Array, x: jax.Array
	) -> jax.Array:

		def score(position: jax.Array):
			return self.posterior_score(position, x)

		return self.posterior_sampler(key, score, z)

	def sample_posterior(
		self, key: jax.Array, z: jax.Array, x: jax.Array, train_idx: int = 0
	) -> jax.Array:
		self.eval()
		return self._sample_posterior(key, z, x)

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		num_samples = x.shape[0]
		contrastive_div = self.ebm.loss(z_post, z_prior) / num_samples
		recon = self.gen.loss(x, z_post) / num_samples
		return contrastive_div + recon

	def adapt_temps(self, train_idx: int, num_updates: int) -> None:
		pass


class mleEBM(MLE, neuralEBM):
	def __init__(self, config, rngs):
		super().__init__(config, rngs)
		self.mcmc_setup(config.gen.mcmc)


class mleKAEM(MLE, KAEM):
	def __init__(self, config, rngs):
		super().__init__(config, rngs)
		self.mcmc_setup(config.gen.mcmc)
