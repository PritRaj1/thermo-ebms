import jax
import numpy as np
from flax import nnx
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from ..config import KAEMConfig


class chebyKAN(nnx.Module):
	"""1D Chebyshev polynomial latent density function"""

	def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):
		self.mixture = config.mixture
		self.sigma = config.p0_stddev

		# Kolmogorov-Arnold Theorem width choices, n -> 2n+1
		self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
		self.P = P

		self.degree = config.degree
		self.coeff = nnx.Param(rngs.normal((1, self.degree + 1, self.Q, P)))
		self.w_cheby = nnx.Param(rngs.normal((1, 1, self.Q, P)))
		self.w_base = nnx.Param(rngs.normal((1, 1, self.Q, P)))

		# Mixture component to sample
		self.reg = config.mixture_regularization
		self.alpha = nnx.Param(jnp.ones((1, 1, self.Q, P))) if self.mixture else None
		self.component = (
			nnx.Variable(jnp.arange(self.Q)[None, None, :, None])
			if self.mixture
			else None
		)

		# Gauss–Legendre quadrature for Inverse Transform
		self.numquad = config.numquad
		self.update_every = config.domain_update_freq
		lo = jnp.full((1, 1, 1, self.P), -1.0)
		hi = jnp.full((1, 1, 1, self.P), 1.0)
		nodes, weights = self.adapt_gauss(lo, hi)
		self.nodes = nnx.Variable(nodes)
		self.weights = nnx.Variable(weights)

	def chebyshev(
		self,
		z: jax.Array,
		coeff: jax.Array,
		w_cheby: jax.Array,
		w_base: jax.Array,
	) -> jax.Array:
		z_cheby = nnx.tanh(z)
		T = [jnp.ones_like(z_cheby), z_cheby]
		for i in range(2, self.degree + 1):
			T.append(2 * z_cheby * T[-1] - T[-2])
		cheby = jnp.sum(coeff * jnp.concat(T, axis=1), axis=1, keepdims=True)
		return w_cheby * cheby + w_base * nnx.hard_swish(z)

	def expand_p(self, x: np.ndarray) -> jax.Array:
		return jnp.repeat(
			jnp.expand_dims(jnp.array(x), axis=1),
			self.P,
			axis=1,
		).reshape(self.numquad, 1, 1, self.P)

	def adapt_gauss(self, lo: jax.Array, hi: jax.Array) -> tuple[jax.Array, jax.Array]:
		"""Adapt Gauss-Legendre integration domain"""
		nodes, weights = leggauss(self.numquad)
		nodes, weights = jnp.array(nodes), jnp.array(weights)
		nodes, weights = self.expand_p(nodes), self.expand_p(weights)

		nodes = 0.5 * (hi - lo) * nodes + 0.5 * (lo + hi)
		weights = weights * 0.5 * (hi - lo)
		return nodes, weights

	def gaussleg(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
		mean = jnp.mean(z, axis=0, keepdims=True)
		std = jnp.std(z, axis=0, keepdims=True)
		lo, hi = mean - 3 * std, mean + 3 * std
		nodes, weights = self.adapt_gauss(lo, hi)
		return nodes, weights

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
	) -> jax.Array:
		return self.chebyshev(z, self.coeff, self.w_cheby, self.w_base)

	def en(self, z: jax.Array) -> jax.Array:
		f = self(z)
		if not self.mixture:
			return f.sum()

		f = f + nnx.log_softmax(self.alpha, axis=-2) + self.log_p0(z)
		nodes, weights = self.gaussleg(z)
		Z = jnp.sum(
			weights * jnp.exp(self(nodes) + self.log_p0(nodes)),
			axis=0,
			keepdims=True,
		)

		return nnx.logsumexp(f - jnp.log(Z), axis=-2).sum()

	def prior_score(self, z: jax.Array) -> jax.Array:
		grad_f = jax.grad(self.en)(z)
		if self.mixture:
			return grad_f

		return -grad_f - z / (self.sigma**2)

	def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
		if not self.mixture:
			return self.en(z_post) - self.en(z_prior)

		return -self.en(z_post) + self.reg * jnp.sum(jnp.abs(self.alpha))

	def componentwise_f(self, z: jax.Array) -> jax.Array:
		"""
		In: (numsamples, 1, Q, P)
		Out: (numsamples, 1, 1, P) if mixture else (numsamples, 1, Q, P))
		"""
		coeff = self.select_component(self.coeff)
		w_cheby = self.select_component(self.w_cheby)
		w_base = self.select_component(self.w_base)
		z = self.select_component(z)
		return self.chebyshev(z, coeff, w_cheby, w_base)

	def pdf_per_node(self):
		"""Returns normalized pdf per density"""
		f = jax.vmap(self.componentwise_f)(
			jnp.expand_dims(jnp.repeat(self.nodes, self.Q, axis=-2), axis=1)
		)
		return self.weights * jnp.exp(f.squeeze(axis=2) + self.log_p0(self.nodes))
