import blackjax
import jax
import jax.numpy as jnp
from flax import nnx

from ..config import MCMCConfig, ScoreFn, ThermoConfig, XchangeFn


class mcmc_sampler(nnx.Module):
	def __init__(
		self,
		score: ScoreFn,
		config: MCMCConfig,
		xchange_conf: ThermoConfig | None = None,
	):
		self.eta = config.stepsize
		self.run_iters = config.numsteps
		self.kernel = blackjax.sgld(score)
		self.xchange_every = -1

		if xchange_conf is not None:
			self.xchange_every = (
				xchange_conf.xchange_every if xchange_conf.num_temps > 1 else -1
			)

	def __call__(
		self,
		key: jax.Array,
		z0: jax.Array,
		x: jax.Array | None = None,
		xchange_func: XchangeFn | None = None,
	):
		xchange_bool = (self.xchange_every > 0) and (xchange_func is not None)
		key, runkey = jax.random.split(key)

		def step(carry, idx):
			z, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			z = self.kernel.step(subkey, z, x, self.eta)

			if xchange_bool:

				def swap(s):
					return xchange_func(newkey, s, idx)

				z = jax.lax.cond(
					(idx % self.xchange_every == 0),
					swap,
					lambda s: s,
					z,
				)

			return (z, newkey), None

		(z0, _), _ = jax.lax.scan(step, (z0, runkey), xs=jnp.arange(self.run_iters))
		return z0
