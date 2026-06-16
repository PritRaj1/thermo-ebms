import jax
from flax import nnx
from ml_collections import ConfigDict
from jaxkan import layers

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
		P = config.model.z_dim
		Q = (P - 1) // 2 if config.kan_prior.mixture else 2 * P + 1
		self.ebm.f = kan(n_in=P, n_out=Q, **params)
		self.ebm.en = self.energy

	def energy(self, z: jax.Array) -> jax.Array():
		return self.ebm.f(z).sum()
