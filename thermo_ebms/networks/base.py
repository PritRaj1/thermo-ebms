import jax
from functools import partial
from flax import nnx
from ml_collections import ConfigDict

from .ebm_fcnn import EBM
from .gen_cnn import GEN
from .sampling import NUTS_sampler


class neuralEBM(nnx.Module):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		self.z_dim = config.model.z_dim
		self.prior_sampler = NUTS_sampler(config.ebm)
		self.posterior_sampler = NUTS_sampler(config.gen, config.thermo)
		self.ebm = EBM(config.ebm, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs)
		self.train_idx = 0

	def nuts_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		z0 = jax.random.normal(subkey, (N, 1, 1, self.z_dim)) * self.ebm.sigma
		return z0, key

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		z0, key = self.nuts_init(key, N)
		return self.prior_sampler(key, self.ebm.logprior, z0)

	@partial(nnx.jit, static_argnames=("N",))
	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		z = self.sample_prior(key, N)
		return self.gen(z)
