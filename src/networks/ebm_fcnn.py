import jax
from flax import nnx
from ml_collections import ConfigDict


class EBM(nnx.Module):
	def __init__(self, ebm_config: ConfigDict, rngs: nnx.Rngs):
		in_dim = ebm_config.z_dim
		hidden_dim = ebm_config.energy_dim
		leak_coef = ebm_config.leakyrelu_leak
		self.sigma = ebm_config.p0_stddev

		act = lambda x: nnx.leaky_relu(x, negative_slope=leak_coef)

		self.f = nnx.Sequential(
			nnx.Linear(in_dim, hidden_dim),
			act,
			nnx.Linear(hidden_dim, hidden_dim),
			act,
			nnx.Linear(hidden_dim, out_dim),
		)

	def __call__(self, z: jax.Array) -> jax.Array:
		return self.f(z)

	def logprior(self, z: jax.Array) -> jax.Array:
		"""log(p_α(z)) ∝ f(z) - 0.5 * (z^2) / (σ^2)"""
		return self(z) - 0.5 * (z**2) / (self.sigma**2)
