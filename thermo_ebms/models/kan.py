import jax
import numpy as np
from flax import nnx
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from ..config import KAEMConfig


class wavKAN(nnx.Module):
	"""1D Morlet wavelet latent density function"""

	def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):
		self.mixture = config.mixture
		self.sigma = config.p0_stddev

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

		# Gauss–Legendre quadrature for Inverse Transform
		self.numquad = config.numquad
		nodes, weights = self.adapt_gauss()
		self.nodes = nnx.Variable(nodes)
		self.weights = nnx.Variable(weights)

	def expand_p(self, x: np.ndarray) -> jax.Array:
		return jnp.repeat(
			jnp.expand_dims(jnp.array(x), axis=1),
			self.P,
			axis=1,
		).reshape(self.numquad, 1, 1, self.P)

	def adapt_gauss(
		self, domain: tuple | None = (-1.2, 1.2)
	) -> tuple[jax.Array, jax.Array]:
		"""Adapt Gauss-Legendre integration domain"""
		nodes, weights = leggauss(self.numquad)
		nodes, weights = jnp.array(nodes), jnp.array(weights)

		a, b = domain if domain else (-1.2, 1.2)
		nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
		weights = weights * 0.5 * (b - a)
		return self.expand_p(nodes), self.expand_p(weights)

	def log_p0(self, z: jax.Array) -> jax.Array:
		"""π_0(z) = N(0, 1), in_shape = (N_quad, Q, P)"""
		return (
			-0.5 * (z / self.sigma) ** 2
			- jnp.log(self.sigma)
			- 0.5 * jnp.log(2.0 * jnp.pi)
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

	def __call__(
		self,
		z: jax.Array,
		translation: jax.Array,
		bandwidth: jax.Array,
		tau: jax.Array,
		w_wav: jax.Array,
		w_base: jax.Array,
	) -> jax.Array:
		z_scaled = (z - translation) / bandwidth
		real = jnp.cos(tau * z_scaled)
		envelope = jnp.exp(-(z_scaled**2) / 2)
		return w_wav * (real * envelope) + w_base * nnx.hard_swish(z)

	def en(self, z: jax.Array, partition: jax.Array | None = None) -> jax.Array:
		f = self(z, self.translation, self.bandwidth, self.tau, self.w_wav, self.w_base)
		if not self.mixture:
			return f.sum()

		f = f + nnx.log_softmax(self.alpha, axis=-2) + self.log_p0(z)
		if partition is not None:
			f = f - partition

		return nnx.logsumexp(f, axis=-2).sum()

	def prior_score(self, z: jax.Array) -> jax.Array:
		return jax.grad(self.en)(z)

	def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
		if not self.mixture:
			return -(self.en(z_post) - self.en(z_prior))

		quad = self(
			jnp.repeat(self.nodes, self.Q, axis=-2),
			self.translation,
			self.bandwidth,
			self.tau,
			self.w_wav,
			self.w_base,
		)
		Z = jnp.sum(
			self.weights * jnp.exp(quad + self.log_p0(self.nodes)),
			axis=0,
			keepdims=True,
		)
		return -self.en(z_post, partition=Z)

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

		return self(z, translation, bandwidth, tau, w_wav, w_base)
