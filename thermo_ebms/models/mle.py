import jax
from flax import nnx

from .base import neuralEBM
from .kaem import KAEM


class MLE:
	@nnx.jit
	def _sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		z0, key = self.mcmc_init(key, x.shape[0])

		def score(z: jax.Array) -> jax.Array:
			return self.gen.llhood_score(z, x) + self.ebm.prior_score(z)

		return self.posterior_sampler(key, score, z0)

	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		self.eval()
		return self._sample_posterior(key, x)

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		num_samples = x.shape[0]
		contrastive_div = self.ebm.loss(z_post, z_prior) / num_samples
		recon = self.gen.loss(x, z_post) / num_samples
		return contrastive_div + recon

	def adapt_temps(self, x: jax.Array, z: jax.Array) -> None:
		pass


class mleEBM(MLE, neuralEBM):
	def __init__(self, config, rngs):
		super().__init__(config, rngs)


class mleKAEM(MLE, KAEM):
	def __init__(self, config, rngs):
		super().__init__(config, rngs)
