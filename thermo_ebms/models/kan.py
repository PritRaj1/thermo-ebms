import jax
import jax.numpy as jnp
from flax import nnx

from ..config import KAEMConfig


class rbfKAN(nnx.Module):
	"""1D RBF latent density function"""

	def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):

		self.mixture = config.mixture

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		n_centres = config.numcentres
		centers = jnp.linspace(-1.2, 1.2, n_centres)[None, :, None, None]
		centres = jnp.repeat(jnp.repeat(centers, self.Q, axis=-2), P, axis=-1)
		self.centres = nnx.Param(centres)

		spacing = 2.0 / (n_centres - 1)
		log_var = jnp.full((1, n_centres, self.Q, P), jnp.log(spacing))
		self.log_var = nnx.Param(log_var + rngs.normal((1, n_centres, self.Q, P)))

		self.k = nnx.Param(rngs.normal((1, n_centres, self.Q, P)))
		self.w_rbf = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.w_base = nnx.Param(rngs.normal((1, 1, self.Q, P)))

		# Mixture component to sample
		self.component = nnx.Variable(jnp.arange(self.Q)[None, None, :, None])

	def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
		"""Sample uniformly from Categorical(1:mixture_components`. Called outside JIT"""
		if self.mixture:
			key, subkey = jax.random.split(key)
			self.component.set_value(
				jax.random.randint(
					subkey,
					shape=(N, 1, 1, self.P),
					minval=0,
					maxval=self.Q,
				)
			)

		return key

	def select_component(self, x: jax.Array) -> jax.Array:
		"""Choose mixture component along Q dim"""
		return jnp.take_along_axis(x, self.component, axis=-2)

	def __call__(self, z: jax.Array, sampling: bool = False) -> jax.Array:
		"""
		In: (numsamples, 1, Q, P)
		Out: (numsamples, 1, 1, P) if mixture else (numsamples, 1, Q, P))
		"""
		centres = self.select_component(self.centres)
		log_var = self.select_component(self.log_var)
		k = self.select_component(self.k)
		w_rbf = self.select_component(self.w_rbf)
		w_base = self.select_component(self.w_base)
		if z.shape[-2] > 1:
			z = self.select_component(z)

		var = nnx.softplus(log_var) + 1e-12
		rbf = jnp.exp(-0.5 * (z - centres) ** 2 / var)
		rbf = jnp.sum(rbf * k, axis=1, keepdims=True)
		return w_rbf * rbf + w_base * nnx.hard_swish(z)
