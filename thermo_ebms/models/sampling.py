import jax
import jax.numpy as jnp
from flax import nnx

from ..config import ScoreFn, ULAConfig, XchangeFn


class ula_sampler(nnx.Module):
	"""Unadjusted Langevin Algorithm sampler"""

	def __init__(self, config: ULAConfig):
		self.eta = config.stepsize
		self.run_iters = config.numsteps

	def __call__(
		self,
		key: jax.Array,
		score: ScoreFn,
		z0: jax.Array,
		xchange_func: XchangeFn | None = None,
	):
		key, runkey = jax.random.split(key)

		def step(carry, idx):
			z, newkey = carry
			newkey, subkey, swapkey = jax.random.split(newkey, 3)
			eps = jax.random.normal(subkey, z.shape)
			z = z + self.eta * score(z) + jnp.sqrt(2 * self.eta) * eps
			if xchange_func is not None:
				z = xchange_func(swapkey, z, idx)

			return (z, newkey), None

		(z0, _), _ = jax.lax.scan(step, (z0, runkey), xs=jnp.arange(self.run_iters))
		return z0
