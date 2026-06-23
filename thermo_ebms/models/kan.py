import jax
import jax.numpy as jnp
from flax import nnx
from jaxkan import layers

from ..config import KANConfig

BASES = {
	"rbf": layers.RBFLayer,
	"spline": layers.SplineLayer,
	"chebyshev": layers.ChebyshevLayer,
	"fourier": layers.FourierLayer,
}


class kanBANK(nnx.Module):
	"""KAN module with no inner sum"""

	def __init__(self, config: KANConfig, mixture: bool, P: int, seed0: int):
		self.mixture = mixture

		if config.basis not in BASES:
			raise ValueError(f"Unknown jaxkan basis: {config.basis}")

		basis_cfg = getattr(config, config.basis)
		if basis_cfg is None:
			raise ValueError(f"Missing config for basis: {config.basis}")

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		kan = BASES[config.basis]
		params = dict(basis_cfg)
		self.layers = nnx.List(
			[kan(n_in=1, n_out=self.Q, seed=seed0 + k, **params) for k in range(P)]
		)

		self.numgrid = config.grid_updating.numgrid
		self.freq = config.grid_updating.update_frequency
		self.decay = config.grid_updating.frequency_decay

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

	@nnx.jit
	def _static_update(self, z: jax.Array, layers: nnx.Module):
		z = jnp.reshape(z, (-1, self.P))
		for i in range(len(self.layers)):
			layers[i].update_grid(z[:, i : i + 1], self.numgrid)

		return layers

	def update_grid(self, z: jax.Array, train_idx: int) -> None:
		if train_idx % self.freq == 0:
			self.layers = self._static_update(z, self.layers)

			if train_idx > 1:
				self.freq = jnp.floor(self.freq * (2 - self.decay))  # Decay
