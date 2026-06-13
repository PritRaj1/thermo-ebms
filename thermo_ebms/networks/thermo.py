import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict
import blackjax

from .base import neuralEBM
from .sampling import NUTS_sampler


class thermoEBM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		self.num_temps = config.thermo.num_temps
		self.temps = jnp.linspace(0, 1, self.num_temps)

	def expanded_ll(self, x: jax.Array, z: jax.Array) -> jax.Array:
		x_gen = self.gen(z).reshape(self.num_temps, x.shape[1], *x.shape[2:])
		return ((x - x_gen) ** 2).sum(axis=(2, 3, 4)) / (2 * self.gen.sigma**2)

	def adapt_temps(self, x: jax.Array, z: jax.Array) -> None:
		"""
		Adapt temps by minimising/equalizing KL div between adjacent power posteriors

		KL(p_t || p_{t+Δt}) = 0.5 Var_t[ log p_β(x | z) * Δt^2]
		"""

		rho = self.expanded_ll(x, z).std(axis=1)
		cdf = jnp.cumsum(rho)
		cdf = cdf / cdf[-1]
		self.temps = jnp.interp(jnp.linspace(0, 1, self.num_temps), cdf, self.temps)

	def sample_posterior(self, key, x):
		self.eval()
		x = jnp.expand_dims(x, 0)
		z0, key = self.nuts_init(key, x.shape[1] * self.num_temps)

		def log_powerpost(z: jax.Array) -> jax.Array:
			ll = self.expanded_ll(x, z).mean(axis=1)
			return (self.temps * ll).sum() + self.ebm.logprior(z)

		z_post = self.posterior_sampler(key, log_powerpost, z0)
		self.adapt_temps(x, z_post)
		return z_post.reshape(self.num_temps, x.shape[1], *z0.shape[1:])

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""
		Thermodynamic integration with trapezoidal rule

		1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
		"""
		delta_t = self.temps[1:] - self.temps[:-1]

		def pixel_loss(z_i: jax.Array) -> jax.Array:
			return self.gen.loss(x, z_i)

		expectations = jax.vmap(pixel_loss)(z_post)
		trapz = delta_t * (expectations[1:] + expectations[:-1])
		return 0.5 * trapz.sum()
