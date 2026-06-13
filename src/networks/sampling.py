import jax
from flax import nnx
import blackjax
from ml_collections import ConfigDict
from collections.abc import Callable


class NUTS_sampler(nnx.Module):
	def __init__(self, config: ConfigDict):
		self.warmup = config.nuts_burn_in
		self.iters = config.nuts_numsteps

	def __call__(
		self, key: jax.Array, logprob: Callable[[jax.Array], jax.Array], z0: jax.Array
	):
		key, subkey = jax.random.split(key)
		warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
		(state, params), _ = warmup.run(subkey, z0, num_steps=self.warmup)
		kernel = blackjax.nuts(logprob, **params)

		def step(carry, _):
			st, newkey = carry
			newkey, subkey = jax.random.split(newkey)
			st, _ = kernel.step(subkey, st)
			return (st, newkey), None

		(state, _), _ = jax.lax.scan(step, (state, key), xs=None, length=self.iters)
		return state.position
