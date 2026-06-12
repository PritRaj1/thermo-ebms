import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict

from .base import latentEBM


class mleEBM(latentEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)

	def posterior_score(self, z: jax.Array, x: jax.Array) -> jax.Array:
		"""Returns ∇_z log p_θ(z | x) = ∇_z( log p_β(x | z) * p_α(z) )"""
		grad_ll = jax.grad(lambda zz: self.gen.loglkhood(zz, x))(z)
		return self.prior_score(z) + grad_ll

	def sample_posterior(self, key: jax.Array, x: jax.Array) -> jax.Array:
		z0, key = self.ula_init(key, x.shape[0])
		self.eval()

		def step(carry, _):
			z, key = carry
			key, noise_key = jax.random.split(key)

			score = self.posterior_score(z, x)
			noise = jax.random.normal(noise_key, z.shape)

			eta = self.ula_step_post
			z = z + eta * score + jnp.sqrt(2 * eta) * noise

			return (z, key), None

		(z, _), _ = jax.lax.scan(
			step,
			(z0, key),
			xs=None,
			length=self.ula_iters_post,
		)

		return z


def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
	return self.gen.loss(x, z_post)
