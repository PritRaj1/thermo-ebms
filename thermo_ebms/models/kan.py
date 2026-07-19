import jax
import jax.numpy as jnp
from flax import nnx

from ..config import KAEMConfig


class wavKAN(nnx.Module):
	"""1D Morlet wavelet latent density function"""

	def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):
		self.mixture = config.mixture

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		self.translation = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.bandwidth = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.tau = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.w_wav = nnx.Param(rngs.normal((1, 1, self.Q, P)))
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

	def basis(
		self, z: jax.Array, translation: jax.Array, bandwidth: jax.Array, tau: jax.Array
	) -> jax.Array:
		z = (z - translation) / bandwidth
		real = jnp.cos(tau * z)
		envelope = jnp.exp(-(z**2) / 2)
		return real * envelope

	def __call__(self, z: jax.Array) -> jax.Array:
		wav = self.basis(z, self.translation, self.bandwidth, self.tau)
		f = self.w_wav * wav + self.w_base * nnx.hard_swish(z)
		if not self.mixture:
			return f

		log_alpha = nnx.log_softmax(self.alpha, axis=-2)
		return nnx.logsumexp(f + log_alpha, axis=-2)

	def componentwise_pdf(self, z: jax.Array) -> jax.Array:
		"""
		In: (numsamples, 1, Q, P)
		Out: (numsamples, 1, 1, P) if mixture else (numsamples, 1, Q, P))
		"""
		translation = self.select_component(self.translation)
		bandwidth = self.select_component(self.bandwidth)
		tau = self.select_component(self.tau)
		w_wav = self.select_component(self.w_wav)
		w_base = self.select_component(self.w_base)
		if z.shape[-2] > 1:
			z = self.select_component(z)

		wav = self.basis(z, translation, bandwidth, tau)
		return w_wav * wav + w_base * nnx.hard_swish(z)
