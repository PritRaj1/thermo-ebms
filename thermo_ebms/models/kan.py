import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
from jaxkan import layers

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


class kanBANK(nnx.Module):
	"""KAN module with no inner sum"""

	def __init__(self, config: ConfigDict, P: int, Q: int):
		kan = BASES[config.kan_prior.basis]
		params = {
			k: config.kan_prior[k]
			for k in CONFIG_KEYS[config.kan_prior.basis]
			if k in config.kan_prior
		}
		self.layers = nnx.List(
			[
				kan(n_in=1, n_out=Q, seed=config.model.seed + k, **params)
				for k in range(P)
			]
		)

	def __call__(self, z: jax.Array) -> jax.Array:
		outs = []
		for i, layer in enumerate(self.layers):
			outs.append(layer(z[:, i : i + 1]))
		return jnp.stack(outs, axis=-1)
