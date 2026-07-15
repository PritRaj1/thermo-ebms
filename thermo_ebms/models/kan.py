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
		self.alpha = nnx.Param(rngs.normal((1, 1, self.Q, P))) if self.mixture else None
		self.component = (
			nnx.Variable(jnp.arange(self.Q)[None, None, :, None])
			if self.mixture
			else None
		)

	def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
		"""Sample uniformly from Categorical(1:mixture_components). Called outside JIT"""
		if self.mixture:
			key, subkey = jax.random.split(key)
			self.component.set_value(
				jax.random.categorical(
					subkey,
					logits=self.alpha,
					axis=-2,
					shape=(N, 1, 1, self.P),
				)
			)

		return key

	def select_component(self, x: jax.Array) -> jax.Array:
		"""Choose mixture component along Q dim"""
		if not self.mixture:
			return x

		return jnp.take_along_axis(x, self.component, axis=-2)

	def basis(self, z: jax.Array, coeff: jax.Array) -> jax.Array:
		z = nnx.hard_tanh(z)
		T = [jnp.ones_like(z), z]
		for i in range(2, self.degree + 1):
			T.append(2 * z * T[-1] - T[-2])

		basis = jnp.concat(T, axis=1)
		return jnp.sum(coeff * basis, axis=1, keepdims=True)

	def __call__(self, z: jax.Array) -> jax.Array:
		cheby = self.basis(z, self.coeff)
		f = self.w_cheby * cheby + self.w_base * nnx.hard_swish(z)
		if not self.mixture:
			return f

		log_alpha = nnx.log_softmax(self.alpha, axis=-2)
		return nnx.logsumexp(f - log_alpha, axis=-2)

	def componentwise_pdf(self, z: jax.Array) -> jax.Array:
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

		cheby = self.basis(z, coeff)
		return w_cheby * cheby + w_base * nnx.hard_swish(z)
