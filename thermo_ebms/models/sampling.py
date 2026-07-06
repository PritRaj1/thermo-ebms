import jax
import blackjax
from flax import nnx
import jax.numpy as jnp

from ..config import MCMCConfig, ThermoConfig, ScoreFn, XchangeFn


class mcmc_sampler(nnx.Module):
	def __init__(self, config: MCMCConfig, xchange_conf: ThermoConfig | None = None):
		self.eta = config.stepsize
		self.run_iters = config.numsteps
		self.L = config.numleapfrog
		self.alpha = config.alpha
		self.beta = config.beta
		self.xchange_every = -1

		if xchange_conf is not None:
			self.xchange_every = (
				xchange_conf.xchange_every if xchange_conf.num_temps > 1 else -1
			)

	def __call__(
		self,
		key: jax.Array,
		score: ScoreFn,
		z0: jax.Array,
		xchange_func: XchangeFn | None = None,
		minibatch: jax.Array | None = None,
	):
		xchange_bool = (self.xchange_every > 0) and (xchange_func is not None)
		key, runkey = jax.random.split(key)
		kernel = blackjax.sghmc(
			grad_estimator=score,
			num_integration_steps=self.L,
			alpha=self.alpha,
			beta=self.beta,
		)
		state = kernel.init(z0)

		def step(carry, idx):
			st, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			st = kernel.step(subkey, st, minibatch=minibatch, step_size=self.eta)

			if xchange_bool:

				def swap(s):
					return xchange_func(newkey, s, idx)

				st = jax.lax.cond(
					(idx % self.xchange_every == 0),
					swap,
					lambda s: s,
					st,
				)

			return (st, newkey), None

		(state, _), _ = jax.lax.scan(
			step, (state, runkey), xs=jnp.arange(self.run_iters)
		)
		return state
