import jax
from flax import nnx
import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from .base import neuralEBM
from .kan import wavKAN
from ..config import ModelConfig


def search_one(cdf_1d: jax.Array, u_1d: jax.Array) -> jax.Array:
	return jnp.searchsorted(cdf_1d, u_1d, side="right")


class KAEM(neuralEBM):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		del self.ebm.f

		# No-inner-sum KAN (Q*P 1D functions)
		self.kan = wavKAN(config.kaem, self.z_dim, rngs)
		self.ebm.f = self.kan

		# Gauss–Legendre quadrature for Inverse Transform
		self.numquad = config.kaem.numquad
		nodes, weights = self.adapt_gauss()
		self.nodes = nnx.Variable(nodes)
		self.weights = nnx.Variable(weights)

	def expand_p(self, x: np.ndarray) -> jax.Array:
		return jnp.repeat(
			jnp.expand_dims(jnp.array(x), axis=1),
			self.z_dim,
			axis=1,
		).reshape(self.numquad, 1, 1, self.z_dim)

	def adapt_gauss(
		self, domain: tuple | None = (-1.2, 1.2)
	) -> tuple[jax.Array, jax.Array]:
		"""Adapt Gauss-Legendre integration domain"""
		nodes, weights = leggauss(self.numquad)
		nodes, weights = jnp.array(nodes), jnp.array(weights)

		a, b = domain if domain else (-1.2, 1.2)
		nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
		weights = weights * 0.5 * (b - a)
		return self.expand_p(nodes), self.expand_p(weights)

	def mcmc_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		inner_dim = 1 if self.kan.mixture else self.kan.Q
		z0 = jax.random.normal(subkey, (N, 1, inner_dim, self.z_dim)) * self.ebm.sigma
		return z0, key

	def log_p0(self) -> jax.Array:
		"""π_0(z) = N(0, 1), in_shape = (N_quad, Q, P)"""
		sigma = self.ebm.sigma
		return (
			-0.5 * (self.nodes / sigma) ** 2
			- jnp.log(sigma)
			- 0.5 * jnp.log(2.0 * jnp.pi)
		)

	def invert_cdf(self, u: jax.Array, cdf: jax.Array) -> jax.Array:
		"""Batched inversion; u: (N, Q, P, 1), cdf: (1, Q, P, G) or (N, Q, P, G)"""
		cdf_flat = cdf.reshape(-1, self.numquad)
		u_flat = u.reshape(-1)
		idx = jax.vmap(search_one)(cdf_flat, u_flat).reshape(u.shape)
		nodes = jnp.broadcast_to(
			jnp.reshape(self.nodes, (1, 1, self.kan.P, self.numquad)),
			(u.shape[0], 1, self.z_dim, self.numquad),
		)

		# Quadrature bin bounds
		idx0 = idx.clip(min=0, max=cdf.shape[-1] - 2)
		idx1 = idx0 + 1
		cdf0 = jnp.take_along_axis(cdf, idx0, axis=-1).squeeze(-1)
		cdf1 = jnp.take_along_axis(cdf, idx1, axis=-1).squeeze(-1)
		z0 = jnp.take_along_axis(nodes, idx0, axis=-1).squeeze(-1)
		z1 = jnp.take_along_axis(nodes, idx1, axis=-1).squeeze(-1)

		# Interpolate within bin
		t = (u.squeeze(-1) - cdf0) / jnp.maximum(cdf1 - cdf0, 1e-12)
		return z0 + t * (z1 - z0)

	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		inner_dim = 1 if self.kan.mixture else self.kan.Q
		f = jax.vmap(self.kan.componentwise_pdf)(
			jnp.expand_dims(jnp.repeat(self.nodes, self.kan.Q, axis=-2), axis=1)
		)  # Returns resulting component per node
		pdf = self.weights * jnp.exp(f.squeeze(axis=2) + self.log_p0())

		# Must broadcast num_samples if univariate. Mixture handles through component
		if not self.kan.mixture:
			pdf = jnp.repeat(pdf, N, axis=1)

		# Cumulative density via Gauss-Legendre integral
		cdf = jnp.cumsum(pdf, axis=0)
		cdf /= cdf[-1, :, :, :] + 1e-12  # Normalize

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, inner_dim, self.z_dim, 1))
		z = self.invert_cdf(u, cdf.transpose(1, 2, 3, 0))
		return z[:, None, :, :]

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.kan.sample_mixture(key, N)
		return self._sample_prior(key, N)

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.kan.sample_mixture(key, N)
		return self._fwd(key, N)
