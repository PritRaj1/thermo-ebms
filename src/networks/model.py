import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict

from .ebm_fcnn import EBM
from .gen_cnn import GEN


class latentEBM(nnx.Module):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		self.z_dim = config.model.z_dim

		self.ula_step_prior = config.ebm.ula_eta
		self.ula_iters_prior = config.ebm.ula_numsteps

		self.ula_step_post = config.gen.ula_eta
		self.ula_iters_post = config.gen.ula_numsteps

		self.ebm = EBM(config.ebm, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs)

	def prior_score(self, z: jax.Array) -> jax.Array:
		"""Returns ∇_z( log[ p_α(x) ] )"""
		return jax.grad(self.ebm.logprior)(z)

	def posterior_score(
		self,
		z: jax.Array,
		x: jax.Array,
		t: jax.Array,
	) -> jax.Array:
		"""Returns ∇_z( log[p_θ(z | x)] ) = ∇_z( log[ p_β(x | z)^t * p_α(z) ] )"""
		grad_ll = jax.grad(lambda zz: self.gen.loglkhood(zz, x, t))(z)
		return self.prior_score(z) + grad_ll

	def ula_prior_step(self, carry, _):
		z, key = carry
		key, noise_key = jax.random.split(key)

		score = self.prior_score(z)
		noise = jax.random.normal(noise_key, z.shape)

		eta = self.ula_step_prior
		z = z + eta * score + jnp.sqrt(2 * eta) * noise

		return (z, key), None

	def ula_post_step(self, carry, inputs):
		(z, key), (x, t) = carry

		key, noise_key = jax.random.split(key)

		score = self.posterior_score(z, x, t)
		noise = jax.random.normal(noise_key, z.shape)

		eta = self.ula_step_post
		z = z + eta * score + jnp.sqrt(2 * eta) * noise

		return (z, key), None

	def ula_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		z0 = jax.random.normal(subkey, (N, 1, 1, self.z_dim)) * self.ebm.sigma
		return z0, key

	def sample_prior(self, key: jax.Array) -> jax.Array:
		z0, key = self.ula_init(key, 1)

		(z, _), _ = jax.lax.scan(
			self.ula_prior_step,
			(z0, key),
			xs=None,
			length=self.ula_iters_prior,
		)

		return z

	def sample_posterior(
		self,
		key: jax.Array,
		x: jax.Array,
		t: jax.Array,
	) -> jax.Array:

		z0, key = self.ula_init(key, x.shape[0])

		(z, _), _ = jax.lax.scan(
			self.ula_post_step,
			(z0, key),
			xs=(x, t),
			length=self.ula_iters_post,
		)

		return z
