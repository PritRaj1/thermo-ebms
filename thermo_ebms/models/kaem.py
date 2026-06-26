import jax
from flax import nnx
import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from .base import neuralEBM
from .kan import kanBANK
from ..config import ModelConfig


def get_gauss(
	layer_p: nnx.Module, P: int, quad_degree: int
) -> tuple[jax.Array, jax.Array]:
	nodes, weights = leggauss(quad_degree)
	nodes, weights = jnp.array(nodes), jnp.array(weights)

	grid = layer_p.grid.item
	a, b = grid.min(), grid.max()
	nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
	weights = weights * 0.5 * (b - a)
	return nodes, weights


def search_one(cdf_1d: jax.Array, u_1d: jax.Array) -> jax.Array:
	return jnp.searchsorted(cdf_1d, u_1d, side="right")


class KAEM(neuralEBM):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		del self.ebm.f
		self.base = "kaem"

		# No-inner-sum KAN (Q*P 1D functions)
		self.ebm.f = kanBANK(
			config.kaem.kan, config.kaem.mixture, self.z_dim, config.seed
		)
		self.ebm.en = self.energy

		# Gauss–Legendre quadrature for Inverse Transform
		def expand_p(x: np.ndarray) -> jax.Array:
			return jnp.repeat(
				jnp.expand_dims(jnp.array(x), axis=1),
				self.z_dim,
				axis=1,
			)

		self.numquad = config.kaem.numquad
		nodes, weights = leggauss(self.numquad)
		self.nodes = nnx.Variable(expand_p(nodes))
		self.weights = nnx.Variable(expand_p(weights))
		self.init_gauss()

		# Mixture component to sample
		self.component = nnx.Variable(jnp.arange(self.ebm.f.Q)[None, :, None])

	def energy(self, z: jax.Array) -> jax.Array():
		return jnp.take_along_axis(self.ebm.f(z), self.component, axis=1).sum()

	def update_grid(self, z: jax.Array, train_idx: int) -> None:
		self.ebm.f.update_grid(z, train_idx)

	def init_gauss(self) -> None:
		"""Adapt Gauss-Legendre integration domain"""
		if hasattr(self.ebm.f.layers[0], "grid"):
			nodes, weights = [], []

			for layer in self.ebm.f.layers:
				n, w = get_gauss(layer, self.z_dim, self.numquad)
				nodes.append(n)
				weights.append(w)

			self.nodes[...] = jnp.stack(nodes, axis=-1)
			self.weights[...] = jnp.stack(weights, axis=-1)

	def log_p0(self, z: jax.Array) -> jax.Array:
		"""π_0(z) = N(0, 1)"""
		sigma = self.ebm.sigma
		return -0.5 * (z / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)

	def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
		"""Sample uniformly from Categorical(1:mixture_components`. Called outside JIT"""
		if self.ebm.f.mixture:
			key, subkey = jax.random.split(key)
			self.component.set_value(
				jax.random.randint(
					subkey,
					shape=(N, 1, self.z_dim),
					minval=0,
					maxval=self.ebm.f.Q,
				)
			)

		return key

	def invert_cdf(self, u: jax.Array, cdf: jax.Array) -> jax.Array:
		"""Batched inversion; u: (N, Q, P, 1), cdf: (1, Q, P, G) or (N, Q, P, G)"""
		cdf_flat = cdf.reshape(-1, cdf.shape[-1])
		u_flat = u.reshape(-1)
		idx = jax.vmap(search_one)(cdf_flat, u_flat).reshape(u.shape)
		nodes = jnp.broadcast_to(
			self.nodes.T[None, None, :, :],
			(u.shape[0], 1, self.z_dim, self.nodes.shape[0]),
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

	@nnx.jit(static_argnames=("N",))
	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		inner_dim = 1 if self.ebm.f.mixture else self.ebm.f.Q

		nodes = self.nodes[:, None, :].repeat(inner_dim, axis=1)  # Broadcast Q
		f = jnp.take_along_axis(
			self.ebm(nodes)[None, :, :, :],  # Unsqueeze num_samples
			self.component[:, None, :, :],  # Unsqueeze N_quad
			axis=2,
		)

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, inner_dim, self.z_dim, 1))

		pdf = self.weights[None, :, None, :] * jnp.exp(
			f + self.log_p0(self.nodes)[None, :, None, :]
		)

		# Cumulative density via Gauss-Legendre integral
		cdf = jnp.cumsum(pdf, axis=1)
		cdf /= cdf[:, -1:, :, :] + 1e-12  # Normalize

		z = self.invert_cdf(u, cdf.transpose(0, 2, 3, 1))

		q = jnp.arange(inner_dim)
		return z[:, None, q, :]

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.sample_mixture(key, N)
		return self._sample_prior(key, N)

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.sample_mixture(key, N)
		return self._fwd(key, N)
