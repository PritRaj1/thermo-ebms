import jax
import jax.numpy as jnp
from flax import nnx
import blackjax
from ml_collections import ConfigDict
from collections.abc import Callable


class NUTS_sampler(nnx.Module):
	def __init__(self, config: ConfigDict, xchange_conf: ConfigDict = None):
		self.warmup = config.nuts_burn_in
		self.iters = config.nuts_numsteps
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
		key, subkey = jax.random.split(key)
		warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
		(state, params), _ = warmup.run(subkey, z0, num_steps=self.warmup)
		kernel = blackjax.nuts(logprob, **params)
		xchange_bool = (self.xchange_every > 0) & (xchange_func is not None)

		def step(carry, idx):
			st, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			st, _ = kernel.step(subkey, st)

			if xchange_bool & (idx % self.xchange_every == 0):
				st.positions = xchange_func(newkey, st.positions, idx)

			return (st, newkey), None

		(state, _), _ = jax.lax.scan(step, (state, key), xs=jnp.arange(self.iters))
		return state.position
