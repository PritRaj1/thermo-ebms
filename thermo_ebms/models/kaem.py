import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config import ModelConfig
from .gen_cnn import GEN
from .kan import KAN
from .sampling import mcmc_sampler


class KAEM(nnx.Module):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		self.z_dim = config.z_dim
		self.posterior_sampler = mcmc_sampler(config.gen.mcmc, config.thermo)
		self.ebm = KAN(config.kaem, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs, sum_latent=not self.ebm.mixture)
		self.num_temps = -1
		self.adapt_temp_freq = -1

	def mcmc_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		inner_dim = 1 if self.ebm.mixture else self.ebm.Q
		z0 = jax.random.normal(subkey, (N, 1, inner_dim, self.z_dim)) * self.ebm.sigma
		return z0, key

	def invert_cdf(self, u: jax.Array, cdf: jax.Array, grid: jax.Array) -> jax.Array:
		"""Batched inversion; u: (N, Q, P, 1), cdf: (1, Q, P, G) or (N, 1, P, G), grid: (1, Q, P, G) or (N, 1, P, G)"""
		cdf_flat = cdf.reshape(-1, self.ebm.numquad)
		u_flat = u.reshape(-1)
		grid_flat = grid.reshape(-1, self.ebm.numquad)
		z = jax.vmap(jnp.interp)(u_flat, cdf_flat, grid_flat)
		return z.reshape(u.shape[0], 1, -1, self.ebm.P)

	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		pdf, grid = self.ebm.pdf_per_node()

		# Must broadcast num_samples if univariate. Mixture handles through component
		if not self.ebm.mixture:
			pdf = jnp.repeat(pdf, N, axis=1)

		# Cumulative density via Gauss-Legendre integral
		cdf = jnp.cumsum(pdf, axis=0)
		cdf = cdf / jnp.maximum(cdf[-1, :, :, :], 1e-12)

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, 1, self.z_dim, 1))
		if not self.ebm.mixture:
			u = jnp.repeat(u, self.ebm.Q, axis=1)

		return self.invert_cdf(u, cdf.transpose(1, 2, 3, 0), grid.transpose(1, 2, 3, 0))

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._sample_prior(key, N)

	@nnx.jit
	def _posterior(self, key: jax.Array, z0: jax.Array, x: jax.Array) -> jax.Array:
		def score(z: jax.Array) -> jax.Array:
			return self.gen.llhood_score(z, x) + self.ebm.prior_score(z)

		return self.posterior_sampler(key, score, z0)

	def adapt_domain(
		self, key: jax.Array, z: jax.Array, x: jax.Array, train_idx: int
	) -> None:
		pass
		# if train_idx % self.ebm.update_every == 0 and train_idx > 0:
		#     if self.ebm.mixture:
		#         z = jnp.repeat(z, self.ebm.Q, axis=-2)
		#         z = self._posterior(key, z, x)
		#
		#     self.ebm.domain_update(z)

	def make_lut(self, lut_size=256) -> np.ndarray:
		"""Returns numpy array for HLS LUT"""
		u = jnp.broadcast_to(
			jnp.expand_dims(jnp.linspace(0.0, 1.0, lut_size), (1, 2, 3)),
			(lut_size, self.ebm.Q, self.ebm.P, 1),
		)

		z_grid = self.ebm.nodes
		sigma = self.ebm.sigma
		log_p0 = (
			-0.5 * (z_grid / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
		)

		pdf = self.ebm.weights * jnp.exp(self.ebm(z_grid) + log_p0)
		cdf = jnp.cumsum(pdf, axis=0)
		cdf = cdf / jnp.maximum(cdf[-1, :, :, :], 1e-12)
		cdf = jnp.repeat(cdf, lut_size, axis=1)
		z_grid = jnp.repeat(z_grid, lut_size, axis=1)
		lut = self.invert_cdf(
			u, cdf.transpose(1, 2, 3, 0), z_grid.transpose(1, 2, 3, 0)
		)
		return np.asarray(lut.reshape(self.ebm.Q, self.ebm.P, lut_size))

	@nnx.jit(static_argnames=("N",))
	def _fwd(self, key: jax.Array, N: int) -> jax.Array:
		key, subkey = jax.random.split(key)
		z = self.sample_prior(subkey, N)
		return self.gen(z), key

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._fwd(key, N)
