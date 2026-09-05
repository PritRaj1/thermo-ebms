import blackjax
import jax
import jax.numpy as jnp
from flax import nnx

from ..config import MCMCConfig, ScoreFn, XchangeFn


class sgld_sampler(nnx.Module):
	"""Stochastic Gradient Langevin Dynamics posterior sampler"""

	def __init__(self, score: ScoreFn, config: MCMCConfig):
		self.eta = config.stepsize
		self.kernel = blackjax.sgld(score)

	def __call__(
		self,
		key: jax.Array,
		z: jax.Array,
		x: jax.Array | None = None,
		xchange_func: XchangeFn | None = None,
	):
		key, stepkey, swapkey = jax.random.split(key, 3)
		z = self.kernel.step(stepkey, z, x, self.eta)

		if xchange_func is not None:
			z = xchange_func(swapkey, z)

		return z


class ula_sampler(nnx.Module):
	"""Unadjusted Langevin Algorithm sampler"""

	def __init__(self, config: MCMCConfig):
		self.eta = config.stepsize
		self.run_iters = config.numsteps

	def __call__(self, key: jax.Array, score: ScoreFn, z0: jax.Array):
		key, runkey = jax.random.split(key)

		def step(carry, idx):
			z, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			eps = jax.random.normal(subkey, z.shape)
			z = z + self.eta * score(z) + jnp.sqrt(2 * self.eta) * eps
			return (z, newkey), None

		(z0, _), _ = jax.lax.scan(step, (z0, runkey), xs=jnp.arange(self.run_iters))
		return z0
