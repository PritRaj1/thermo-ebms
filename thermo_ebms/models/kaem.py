import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
from numpy.polynomial.legendre import leggauss

from .base import neuralEBM
from .kan import kanBANK


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


class KAEM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		del self.ebm.f

		# Kolmogorov-Arnold Theorem width choices
		self.mixture = config.kan_prior.mixture
		self.P = config.model.z_dim
		self.Q = (self.P - 1) // 2 if self.mixture else 2 * self.P + 1

		# No-inner-sum KAN (Q*P 1D functions)
		self.ebm.f = kanBANK(config, self.P, self.Q)
		self.ebm.en = self.energy

		# Gauss–Legendre quadrature for Inverse Transform
		self.numquad = config.kan_prior.numquad
		self.numgrid = config.kan_prior.numgrid
		nodes, weights = leggauss(self.numquad)
		self.nodes, self.weights = (
			jnp.repeat(jnp.expand_dims(jnp.array(nodes), axis=1), self.P, axis=1),
			jnp.repeat(jnp.expand_dims(jnp.array(weights), axis=1), self.P, axis=1),
		)

		self.init_gauss()

	def energy(self, z: jax.Array) -> jax.Array():
		return self.ebm.f(z.squeeze())[self.component].sum()

	def init_gauss(self):
		"""Adapt Gauss-Legendre integration domain"""
		if hasattr(self.ebm.f.layers[0], "grid"):
			nodes, weights = [], []

			for layer in self.ebm.f.layers:
				n, w = get_gauss(layer, self.P, self.numquad)
				nodes.append(n)
				weights.append(w)

			self.nodes = jnp.stack(nodes, axis=-1)
			self.weights = jnp.stack(weights, axis=-1)

	def udpate_grid(self, z: jax.Array) -> None:
		"""KAN grid adaption using least squares"""
		self.ebm.f.kan.update_grid(z, self.numgrid)
		self.init_gauss()

	def log_p0(self, z: jax.Array) -> jax.Array:
		"""π_0(z) = N(0, 1)"""
		sigma = self.ebm.sigma
		return -0.5 * (z / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)

	def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
		"""Sample uniformly from Categorical(1:mixture_components)"""
		key, subkey = jax.random.split(key)
		self.component = (
			jax.random.randint(key, shape=(N, self.P), minval=0, maxval=self.Q)
			if self.mixture
			else jnp.arange(self.Q)
		)
		return key

	def interpolate_bins(self, u_p: jax.Array, cdf_p: jax.Array) -> jax.Array:
		"""Interpolate for each P"""
		return jax.vmap(jnp.interp, in_axes=(0, 1, 1))(u_p, cdf_p, self.nodes)

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		key = self.sample_mixture(key, N)
		f = jnp.take_along_axis(
			self.ebm(self.nodes)[None, :, :, :],
			self.component[:, None, None, :],
			axis=2,
		).squeeze()

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, self.P))

		pdf = jnp.exp(f + self.log_p0(self.nodes)[None, :, :])
		pdf *= self.weights[None, :, :]

		# Cumulative density via Gauss-Legendre integral
		cdf = jnp.cumsum(f, axis=1)
		cdf /= cdf[:, -1:, :] + 1e-12

		z = jax.vmap(self.interpolate_bins, in_axes=(0, 0))(u, cdf)
		return z[:, None, None, :]
