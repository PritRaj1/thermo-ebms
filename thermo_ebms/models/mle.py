import jax
from flax import nnx

from ..config import MCMCConfig
from .base import neuralEBM
from .kaem import KAEM
from .sampling import sgld_sampler


class MLE:
	def posterior_score(self, z: jax.Array, minibatch: jax.Array) -> jax.Array:
		return self.gen.llhood_score(z, minibatch) + self.ebm.prior_score(z)

	def sgld_setup(self, config: MCMCConfig):
		self.posterior_sampler = sgld_sampler(self.posterior_score, config)

	@nnx.jit
	def _sample_posterior(
		self, key: jax.Array, z: jax.Array, x: jax.Array
	) -> jax.Array:
		return self.posterior_sampler(key, z, x=x)

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
	def __init__(self, config, rngs, sgld_correction: int = 1):
		super().__init__(config, rngs, sgld_correction=sgld_correction)
		self.sgld_setup(config.gen.mcmc)


class mleKAEM(MLE, KAEM):
	def __init__(self, config, rngs, sgld_correction: int = 1):
		super().__init__(config, rngs, sgld_correction=sgld_correction)
		self.sgld_setup(config.gen.mcmc)
