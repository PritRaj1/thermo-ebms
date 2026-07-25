import jax
from flax import nnx
import jax.numpy as jnp

from .kan import wavKAN
from .gen_cnn import GEN
from ..config import ModelConfig
from .sampling import mcmc_sampler


def search_one(cdf_1d: jax.Array, u_1d: jax.Array) -> jax.Array:
	return jnp.searchsorted(cdf_1d, u_1d, side="right")


class KAEM(nnx.Module):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		self.z_dim = config.z_dim
		self.posterior_sampler = mcmc_sampler(config.gen.mcmc, config.thermo)
		self.ebm = wavKAN(config.kaem, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs)
		self.num_temps = -1
		self.adapt_temp_freq = -1

	def mcmc_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		inner_dim = 1 if self.ebm.mixture else self.ebm.Q
		z0 = jax.random.normal(subkey, (N, 1, inner_dim, self.z_dim)) * self.ebm.sigma
		return z0, key

	def invert_cdf(self, u: jax.Array, cdf: jax.Array) -> jax.Array:
		"""Batched inversion; u: (N, Q, P, 1), cdf: (1, Q, P, G) or (N, Q, P, G)"""
		cdf_flat = cdf.reshape(-1, self.ebm.numquad + 1)
		u_flat = u.reshape(-1)
		idx = jax.vmap(search_one)(cdf_flat, u_flat).reshape(u.shape)

		min_z = jnp.full((1,) + self.ebm.nodes.shape[1:], self.ebm.domain[0])
		nodes = jnp.concatenate([min_z, self.ebm.nodes], axis=0)
		grid = jnp.broadcast_to(
			jnp.reshape(nodes, (1, 1, self.ebm.P, self.ebm.numquad + 1)),
			(u.shape[0], 1, self.z_dim, self.ebm.numquad + 1),
		)

		# Quadrature bin bounds
		idx0 = idx.clip(min=0, max=cdf.shape[-1] - 2)
		idx1 = idx0 + 1
		cdf0 = jnp.take_along_axis(cdf, idx0, axis=-1).squeeze(-1)
		cdf1 = jnp.take_along_axis(cdf, idx1, axis=-1).squeeze(-1)
		z0 = jnp.take_along_axis(grid, idx0, axis=-1).squeeze(-1)
		z1 = jnp.take_along_axis(grid, idx1, axis=-1).squeeze(-1)

		# Interpolate within bin
		t = (u.squeeze(-1) - cdf0) / jnp.maximum(cdf1 - cdf0, 1e-12)
		return z0 + t * (z1 - z0)

	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		inner_dim = 1 if self.ebm.mixture else self.ebm.Q
		pdf = self.ebm.pdf_per_node()

		# Must broadcast num_samples if univariate. Mixture handles through component
		if not self.ebm.mixture:
			pdf = jnp.repeat(pdf, N, axis=1)

		# Cumulative density via Gauss-Legendre integral
		zeros = jnp.zeros((1,) + pdf.shape[1:])
		cdf = jnp.concatenate([zeros, jnp.cumsum(pdf, axis=0)], axis=0)
		cdf /= cdf[-1, :, :, :] + 1e-12  # Normalize

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, inner_dim, self.z_dim, 1))
		z = self.invert_cdf(u, cdf.transpose(1, 2, 3, 0))
		return z[:, None, :, :]

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._sample_prior(key, N)

	@nnx.jit(static_argnames=("N",))
	def _fwd(self, key: jax.Array, N: int) -> jax.Array:
		key, subkey = jax.random.split(key)
		z = self.sample_prior(key, N)
		return self.gen(z), key

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._fwd(key, N)
