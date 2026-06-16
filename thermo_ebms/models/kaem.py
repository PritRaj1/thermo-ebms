import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
from jaxkan import layers
from numpy.polynomial.legendre import leggauss

from .base import neuralEBM

BASES = {
	"rbf": layers.RBFLayer,
	"spline": layers.SplineLayer,
	"chebyshev": layers.ChebyshevLayer,
	"fourier": layers.FourierLayer,
}

CONFIG_KEYS = {
	"rbf": ["sigma", "centers"],
	"spline": ["k", "G"],
	"chebyshev": ["degree"],
	"fourier": ["modes"],
}


class KAEM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		del self.ebm.f
		kan = BASES[config.kan_prior.basis]
		params = {
			k: config.kan_prior[k] for k in CONFIG_KEYS[kan] if k in config.kan_prior
		}

		# Kolmogorov-Arnold Theorem width choices
		self.mixture = config.kan_prior.mixture
		self.P = config.model.z_dim
		self.Q = (self.P - 1) // 2 if self.mixture else 2 * self.P + 1
		self.ebm.kan = kan(n_in=self.P, n_out=self.Q, **params)
		self.ebm.f = self.ebm.kan.basis
		self.ebm.en = self.energy

		# Gauss–Legendre quadrature for Inverse Transform
		self.quad_degree = config.kan_prior.quadrature_degree
		self.numgrid = config.kan_prior.grid_number
		self.init_gauss()

	def init_gauss(self):
		"""Adapt Gauss-Legendre integration domain"""
		nodes, weights = leggauss(self.quad_degree)
		self.nodes, self.weights = (
			jnp.repeat(jnp.expand_dims(jnp.array(nodes), axis=1), self.P, axis=1),
			jnp.repeat(jnp.expand_dims(jnp.array(weights), axis=1), self.P, axis=1),
		)

		if hasattr(self.ebm.kan, "grid"):
			grid = self.ebm.kan.grid.item
			a, b = grid.min(), grid.max()
			self.nodes = 0.5 * (b - a) * self.nodes + 0.5 * (a + b)
			self.weights = self.weights * 0.5 * (b - a)

	def udpate_grid(self, z: jax.Array) -> None:
		"""KAN grid adaption using least squares"""
		self.ebm.kan.update_grid(z, self.numgrid)
		self.init_gauss()

	def sample_mixture(self, key: jax.Array, N: int) -> None:
		"""Sample uniformly from Categorical(1:mixture_components)"""
		key, subkey = jax.random.split(key)
		self.component = (
			jax.random.randint(key, shape=N, minval=0, maxval=self.P)
			if self.mixture
			else jnp.arange(self.Q)
		)

	def choose_component(self, z: jax.Array) -> jax.Array:
		"""Select mixture component to sample from"""
		return z[jnp.arange(z.shape[0]), :, self.component, :].squeeze()

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		self.sample_mixture()

	def energy(self, z: jax.Array) -> jax.Array():
		return self.ebm.f(self.choose_component(z)).sum()
