import jax
import jax.numpy as jnp
from flax import nnx

from ..config import EBMConfig


class EBM(nnx.Module):
	half_prec: jnp.dtype = jnp.bfloat16
	full_prec: jnp.dtype = jnp.float32

	def __init__(
		self,
		ebm_config: EBMConfig,
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
					nnx.Linear(
						z_dim,
						width,
						rngs=rngs,
						param_dtype=self.full_prec,
						dtype=self.half_prec,
					),
					act,
				]
			)
			z_dim = width

		self.f = nnx.Sequential(*layers)

	def __call__(self, z: jax.Array) -> jax.Array:
		z = z.astype(self.half_prec)
		return self.f(z).astype(self.full_prec)

	def en(self, z: jax.Array) -> jax.Array:
		return self(z).sum()

	def prior_score(self, z: jax.Array) -> jax.Array:
		"""∇_z log(p_α(z)) ∝ ∇_z f(z) - 0.5 * ||z||^2 / σ^2"""
		grad_f = jax.grad(self.en)(z)
		return grad_f - z / (self.sigma**2)

	def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
		if jnp.ndim(z_post) > jnp.ndim(z_prior):
			z_post = z_post[-1, :, :, :]  # Final thermo samples = posterior

		return (self.en(z_post) - self.en(z_prior)) / z_prior.shape[0]
