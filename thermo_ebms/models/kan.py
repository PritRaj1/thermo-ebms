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

	def __init__(self, config: ConfigDict, P: int, seed0: int):
		self.mixture = config.mixture

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		kan = BASES[config.basis]
		params = {k: config[k] for k in CONFIG_KEYS[config.basis] if k in config}
		self.layers = nnx.List(
			[kan(n_in=1, n_out=self.Q, seed=seed0 + k, **params) for k in range(P)]
		)

	def __call__(self, z: jax.Array) -> jax.Array:
		batch = z.shape[0]
		z = jnp.reshape(z, (-1, self.P))

		outs = []
		for i, layer in enumerate(self.layers):
			outs.append(layer(z[:, i : i + 1]))

		# Mixture -> in (B, Q, P) already
		en = jnp.stack(outs, axis=-1)
		if self.mixture:
			return en

		# univariate
		q = jnp.arange(self.Q)
		return en.reshape(batch, self.Q, self.Q, self.P)[:, q, q, :]
