import jax
import jax.numpy as jnp
from flax import nnx

from ..config import KAEMConfig


class chebyKAN(nnx.Module):
	"""1D Chebyshev polynomial latent density function"""

	def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):
		self.mixture = config.mixture

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		self.degree = config.degree
		self.coeff = nnx.Param(rngs.normal((1, self.degree + 1, self.Q, P)))
		self.w_cheby = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.w_base = nnx.Param(rngs.normal((1, 1, self.Q, P)))

		# Mixture component to sample
		self.component = nnx.Variable(jnp.arange(self.Q)[None, None, :, None])

	def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
		"""Sample uniformly from Categorical(1:mixture_components). Called outside JIT"""
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

		T_0 = 1, T_1 = z
		T_n = 2z*T_{n-1} - T_{n-2}
		"""
		coeff = self.select_component(self.coeff)
		w_cheby = self.select_component(self.w_cheby)
		w_base = self.select_component(self.w_base)
		if z.shape[-2] > 1:
			z = self.select_component(z)

		z = nnx.tanh(z)
		T = [jnp.ones_like(z), z]
		for i in range(2, self.degree + 1):
			T.append(2 * z * T[-1] - T[-2])

		basis = jnp.concat(T, axis=1)
		cheby = jnp.sum(coeff * basis, axis=1, keepdims=True)
		return w_cheby * cheby + w_base * nnx.hard_swish(z)
