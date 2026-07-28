import jax
from flax import nnx
import jax.numpy as jnp

from .kan import chebyKAN
from .gen_cnn import GEN
from ..config import ModelConfig
from .sampling import mcmc_sampler


class KAEM(nnx.Module):
	def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
		self.z_dim = config.z_dim
		self.posterior_sampler = mcmc_sampler(config.gen.mcmc, config.thermo)
		self.ebm = chebyKAN(config.kaem, self.z_dim, rngs)
		self.gen = GEN(config.gen, self.z_dim, rngs)
		self.num_temps = -1
		self.adapt_temp_freq = -1

	def mcmc_init(self, key: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
		key, subkey = jax.random.split(key)
		inner_dim = 1 if self.ebm.mixture else self.ebm.Q
		z0 = jax.random.normal(subkey, (N, 1, inner_dim, self.z_dim)) * self.ebm.sigma
		return z0, key

	def invert_cdf(self, u: jax.Array, cdf: jax.Array) -> jax.Array:
		"""Batched inversion; u: (N, Q, P, 1), cdf: (1, Q, P, G) or (N, 1, P, G)"""
		cdf_flat = cdf.reshape(-1, self.ebm.numquad)
		u_flat = u.reshape(-1)
		grid = jnp.repeat(
			jnp.reshape(self.ebm.nodes, (1, 1, self.z_dim, self.ebm.numquad)),
			u.shape[0],
			axis=0,
		)
		z = jax.vmap(jnp.interp)(u_flat, cdf_flat, grid.reshape(-1, self.ebm.numquad))
		return z.reshape(u.shape[0], 1, -1, self.ebm.P)

	def _sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		"""Inverse transform sampling from p_α(z) ∝ exp(f(z)) ⋅ π(Z)"""
		pdf = self.ebm.pdf_per_node()

		# Must broadcast num_samples if univariate. Mixture handles through component
		if not self.ebm.mixture:
			pdf = jnp.repeat(pdf, N, axis=1)

		# Cumulative density via Gauss-Legendre integral
		cdf = jnp.cumsum(pdf, axis=0)
		cdf = cdf / cdf[-1, :, :, :]

		key, subkey = jax.random.split(key)
		u = jax.random.uniform(subkey, shape=(N, 1, self.z_dim, 1))
		if not self.ebm.mixture:
			u = jnp.repeat(u, self.ebm.Q, axis=1)

		return self.invert_cdf(u, cdf.transpose(1, 2, 3, 0))

	def sample_prior(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._sample_prior(key, N)

	@nnx.jit(static_argnames=("N",))
	def _fwd(self, key: jax.Array, N: int) -> jax.Array:
		key, subkey = jax.random.split(key)
		z = self.sample_prior(key, N)
		return self.gen(z), key

	def __call__(self, key: jax.Array, N: int) -> jax.Array:
		self.eval()
		key = self.ebm.sample_mixture(key, N)
		return self._fwd(key, N)

	def update_domain(self, z: jax.Array, step: int) -> None:
		pass
		# if step % self.ebm.update_every == 0 and step > 0:
		#   nodes, weights = self.ebm.gaussleg(z)
		#   self.ebm.nodes[...] = nodes
		#   self.ebm.weights[...] = weights
