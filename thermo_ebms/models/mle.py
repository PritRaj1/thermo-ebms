import jax

from .base import neuralEBM
from .kaem import KAEM


class _MLE:
	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		def score(z: jax.Array) -> jax.Array:
			return self.gen.posterior_score(z, x) + self.ebm.prior_score(z)

		z0, key = self.mcmc_init(key, x.shape[0])
		return self.posterior_sampler(key, score, z0)

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		return self.gen.loss(x, z_post)


class mleEBM(_MLE, neuralEBM):
	pass


class mleKAEM(_MLE, KAEM):
	pass
