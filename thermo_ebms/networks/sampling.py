import jax
import jax.numpy as jnp
from flax import nnx
import blackjax
from ml_collections import ConfigDict
from collections.abc import Callable


class mcmc_sampler(nnx.Module):
	def __init__(self, config: ConfigDict, xchange_conf: ConfigDict = None):
		self.step_size = config.mcmc_stepsize
		self.run_iters = config.mcmc_numsteps
		self.xchange_every = -1

		if xchange_conf is not None:
			self.xchange_every = (
				xchange_conf.xchange_every if xchange_conf.num_temps > 1 else -1
			)

	def __call__(
		self,
		key: jax.Array,
		logprob: Callable[[jax.Array], jax.Array],
		z0: jax.Array,
		xchange_func: Callable[[jax.Array, jax.Array], jax.Array] = None,
	):
		xchange_bool = (self.xchange_every > 0) & (xchange_func is not None)
		key, runkey = jax.random.split(key)
		kernel = blackjax.mala(logprob, self.step_size)
		state = kernel.init(z0)

		def step(carry, idx):
			st, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			st, _ = kernel.step(subkey, st)

			if xchange_bool:

				def swap(s):
					return s._replace(position=xchange_func(newkey, st.position, idx))

				st = jax.lax.cond(
					xchange_bool & (idx % self.xchange_every == 0),
					swap,
					lambda s: s,
					st,
				)

			return (st, newkey), None

		(state, _), _ = jax.lax.scan(
			step, (state, runkey), xs=jnp.arange(self.run_iters)
		)
		return state.position
