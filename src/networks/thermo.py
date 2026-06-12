import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict

from .base import latentEBM


class thermoEBM(latentEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)

		self.train_idx = 0

		self.num_temps = config.thermo.num_temps
		self.p_cycles = config.thermo.annealing_cycles
		self.p_start = config.thermo.p_start
		self.p_end = config.thermo.p_end
		self.num_updates = config.training.epochs

		self.adapt_powerlaw()
		self.adapt_temps()

	def adapt_powerlaw(self):
		t_i = 2.0 * jnp.pi * (self.p_cycles + 0.5) * self.train_idx / self.num_updates
		self.p = self.p_start + (self.p_end - self.p_start) * 0.5 * (1.0 - jnp.cos(t_i))

	def adapt_temps(self):
		self.temps = (jnp.arange(self.num_temps + 1) / self.num_temps) ** self.p

	def posterior_score(self, z, x, t):
		"""Returns ∇_z log p_θ(z | x) = ∇_z( log p_β(x | z)^t * p_α(z) )"""
		grad_ll = t * jax.grad(lambda zz: self.gen.loglkhood(zz, x))(z)
		return self.prior_score(z) + grad_ll

	def _sample_temp(self, key, x, t):
		z0, key = self.ula_init(key, x.shape[0])

		def _ula(carry, _):
			z, key = carry
			key, noise_key = jax.random.split(key)

			score = self.posterior_score(z, x, t)
			noise = jax.random.normal(noise_key, z.shape)

			eta = self.ula_step_post
			z = z + eta * score + jnp.sqrt(2.0 * eta) * noise

			return (z, key), None

		(zT, _), _ = jax.lax.scan(
			_ula,
			(z0, key),
			xs=None,
			length=self.ula_iters_post,
		)

		return zT

	def sample_posterior(self, key, x):

		def single_temp(t):
			k = jax.random.fold_in(key, jnp.asarray(t))
			return self._sample_temp(k, x, t)

		return jax.vmap(single_temp)(self.temps)
