import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
import blackjax

from .base import latentEBM


class mleEBM(latentEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)

	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		self.eval()

		def logpost(z: jax.Array) -> jax.Array:
			return self.gen.loglkhood(z, x) + self.ebm.logprior(z)

		mala_kernel = blackjax.mala(logpost, self.ula_step_post)
		z0, key = self.ula_init(key, x.shape[0])
		state = mala_kernel.init(z0)

		def step(carry, _):
			st, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			st, _ = mala_kernel.step(subkey, st)
			return (st, newkey), None

		(state, _), _ = jax.lax.scan(
			step,
			(state, key),
			xs=None,
			length=self.ula_iters_post,
		)

		return state.position


def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
	return self.gen.loss(x, z_post)
