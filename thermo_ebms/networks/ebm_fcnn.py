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
		dims = ebm_config.layer_widths
		self.sigma = ebm_config.p0_stddev

		def act(x: jax.Array) -> jax.Array:
			return nnx.leaky_relu(x, negative_slope=ebm_config.leakyrelu_leak)

		layers = []
		for width in dims:
			layers.extend(
				[
					nnx.Linear(z_dim, width, rngs=rngs),
					act,
				]
			)
			z_dim = width

		self.f = nnx.Sequential(*layers)

	def __call__(self, z: jax.Array) -> jax.Array:
		return self.f(z)

	def en(self, z: jax.Array) -> jax.Array:
		return self.f(z).sum()

	def prior_score(self, z: jax.Array) -> jax.Array:
		"""∇_z log(p_α(z)) ∝ ∇_z f(z) - 0.5 * ||z||^2 / σ^2"""
		grad_f = jax.grad(self.en)(z)
		return grad_f - 0.5 * (z**2) / (self.sigma**2)

	def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
		return self(z_post).mean() - self(z_prior).mean()
