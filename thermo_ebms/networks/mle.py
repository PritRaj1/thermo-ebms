import jax
from flax import nnx
from ml_collections import ConfigDict

from .base import neuralEBM


class mleEBM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)

	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		def logpost(z: jax.Array) -> jax.Array:
			return self.gen.loglkhood(z, x) + self.ebm.logprior(z)

		z0, key = self.mcmc_init(key, x.shape[0])
		return self.posterior_sampler(key, logpost, z0)

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		return self.gen.loss(x, z_post)
