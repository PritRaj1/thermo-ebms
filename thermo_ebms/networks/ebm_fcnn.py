import jax
from flax import nnx
from ml_collections import ConfigDict


class EBM(nnx.Module):
	def __init__(
		self,
		ebm_config: ConfigDict,
		z_dim: int,
		rngs: nnx.Rngs,
	):
		hidden_dim = ebm_config.energy_dim
		self.sigma = ebm_config.p0_stddev

		def act(x: jax.Array) -> jax.Array:
			return nnx.leaky_relu(x, negative_slope=ebm_config.leakyrelu_leak)

		self.f = nnx.Sequential(
			nnx.Linear(z_dim, hidden_dim, rngs=rngs),
			act,
			nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
			act,
			nnx.Linear(hidden_dim, z_dim, rngs=rngs),
		)

	def __call__(self, z: jax.Array) -> jax.Array:
		return self.f(z)

	def logprior(self, z: jax.Array) -> jax.Array:
		"""log(p_α(z)) ∝ f(z) - 0.5 * ||z||^2 / σ^2"""
		lp = self(z) - 0.5 * (z**2) / (self.sigma**2)
		return lp.sum()

	def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
		return self(z_post).mean() - self(z_prior).mean()
