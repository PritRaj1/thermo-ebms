import jax
from flax import nnx

from ..config import MCMCConfig
from .base import neuralEBM
from .kaem import KAEM
from .sampling import mcmc_sampler


class MLE:
	def sgld_steup(self, config: MCMCConfig):
		self.posterior_sampler = mcmc_sampler(self.posterior_score, config)

	def posterior_score(self, z: jax.Array, minibatch: jax.Array) -> jax.Array:
		return self.gen.llhood_score(z, minibatch) + self.ebm.prior_score(z)

	@nnx.jit
	def _sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		z0, key = self.mcmc_init(key, x.shape[0])
		return self.posterior_sampler(key, z0, x=x)

	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		self.eval()
		return self._sample_posterior(key, x)

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
		self.sgld_steup(config.gen.mcmc)


class mleKAEM(MLE, KAEM):
	def __init__(self, config, rngs):
		super().__init__(config, rngs)
		self.sgld_steup(config.gen.mcmc)
