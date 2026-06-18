import jax
from flax import nnx

from .ebm_dense import EBM
from .gen_cnn import GEN
from .sampling import mcmc_sampler
from ..config import ModelConfig


class neuralEBM(nnx.Module):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		self.z_dim = config.z_dim
		self.prior_sampler = mcmc_sampler(config.ebm.mcmc)
		self.posterior_sampler = mcmc_sampler(config.gen.mcmc, config.thermo)
		self.ebm = EBM(config.ebm, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs)
		self.train_idx = 0
		self.num_temps = -1
		self.base = "neural"

	def mcmc_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		z0 = jax.random.normal(subkey, (N, 1, 1, self.z_dim)) * self.ebm.sigma
		return z0, key

	@nnx.jit(static_argnames=("N",))
	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		z0, key = self.mcmc_init(key, N)
		return self.prior_sampler(key, self.ebm.prior_score, z0)

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		return self._sample_prior(key, N)

	@nnx.jit(static_argnames=("N",))
	def _fwd(self, key: jax.Array, N: int) -> jax.Array:
		key, subkey = jax.random.split(key)
		z = self.sample_prior(key, N)
		return self.gen(z), key

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		return self._fwd(key, N)
