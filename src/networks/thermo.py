import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
import blackjax

from .base import latentEBM
from .sampling import NUTS_sampler


class thermoEBM(latentEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)

		self.num_temps = config.thermo.num_temps
		self.p_cycles = config.thermo.annealing_cycles
		self.p_start = config.thermo.p_start
		self.p_end = config.thermo.p_end

		self.adapt_powerlaw()
		self.adapt_temps()

	def adapt_powerlaw(self):
		t_i = 2.0 * jnp.pi * (self.p_cycles + 0.5) * self.train_idx / self.num_updates
		self.p = self.p_start + (self.p_end - self.p_start) * 0.5 * (1.0 - jnp.cos(t_i))

	def adapt_temps(self):
		self.temps = (jnp.arange(self.num_temps) / self.num_temps) ** self.p

	def _sample_temp(self, key, x, t):
		def log_powerpost(z: jax.Array) -> jax.Array:
			return t * self.gen.loglkhood(z, x) + self.ebm.logprior(z)

		z0, key = self.nuts_init(key, x.shape[0])
		return self.posterior_sampler(key, log_powerpost, z0)

	def sample_posterior(self, key, x):
		self.eval()

		def single_temp(k, t):
			return self._sample_temp(k, x, t)

		keys = jax.random.split(key, len(self.temps[1:]))
		return jax.vmap(single_temp)(keys, self.temps[1:])

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""
		Thermodynamic integration with trapezoidal rule

		1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
		"""
		delta_t = self.temps[1:] - self.temps[:-1]
		z_prior = jnp.expand_dims(z_prior, axis=0)
		z = jnp.concatenate([z_prior, z_post], axis=0)

		def pixel_loss(z_i: jax.Array) -> jax.Array:
			return self.gen.loss(x, z_i)

		expectations = jax.vmap(pixel_loss)(z)
		trapz = delta_t * (expectations[1:] + expectations[:-1])
		return 0.5 * trapz.sum()
