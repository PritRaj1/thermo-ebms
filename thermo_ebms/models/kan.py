import jax
import jax.numpy as jnp
from flax import nnx
from jaxkan.models.KAN import KAN

from ..config import KANConfig


# TODO: jaxkan not initialising, need to implement
class kanBANK(nnx.Module):
	"""KAN module with no inner sum"""

	def __init__(self, config: KANConfig, mixture: bool, P: int, seed0: int):
		self.mixture = mixture

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		self.layers = nnx.List(
			[
				KAN(
					layer_dims=[1, self.Q],
					layer_type=config.basis,
					required_parameters=dict(getattr(config, config.basis)),
					seed=seed0 + k,
				)
				for k in range(P)
			]
		)

		self.numgrid = config.grid_updating.numgrid
		self.freq = nnx.Variable(
			jnp.array(config.grid_updating.update_frequency, dtype=jnp.float32)
		)
		self.decay = config.grid_updating.frequency_decay

	def __call__(self, z: jax.Array) -> jax.Array:
		batch = z.shape[0]
		z = jnp.reshape(z, (-1, self.P, 1))

		outs = [layer(z[:, i, :]) for i, layer in enumerate(self.layers)]

		# Mixture -> in (B, Q, P) already
		outs = jnp.stack(outs, axis=-1)
		if self.mixture:
			return outs

		# univariate
		q = jnp.arange(self.Q)
		return outs.reshape(batch, self.Q, self.Q, self.P)[:, q, q, :]

	def update_grid(self, z: jax.Array, train_idx: int) -> None:
		if train_idx % self.freq == 0:
			z = jnp.reshape(z, (-1, self.P, 1))
			for i in range(len(self.layers)):
				self.layers[i].update_grids(x=z[:, i, :], G_new=self.numgrid)

			if train_idx > 1:
				self.freq[...] = jnp.floor(self.freq[...] * (2 - self.decay))  # Decay
