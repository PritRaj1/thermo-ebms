import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict

from .base import neuralEBM


class thermoEBM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		self.num_temps = config.thermo.num_temps
		self.temps = jnp.linspace(0.0, 1.0, self.num_temps)

		# DEO exchange
		self._i_even = jnp.arange(0, self.num_temps - 1, 2)
		self._j_even = self._i_even + 1
		self._i_odd = jnp.arange(1, self.num_temps - 1, 2)
		self._j_odd = self._i_odd + 1

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
		self.temps = jnp.interp(
			jnp.linspace(0, 1, self.num_temps),
			cdf,
			self.temps,
		)

	def replica_xchange(
		self,
		key: jax.Array,
		z: jax.Array,
		step_idx: jax.Array,
		x: jax.Array,
	) -> jax.Array:

		ll = self.expanded_ll(x, z).mean(axis=1)
		use_even = (step_idx % 2) == 0
		i = jax.lax.select(use_even, self._i_even, self._i_odd)
		j = jax.lax.select(use_even, self._j_even, self._j_odd)

		key, subkey = jax.random.split(key)
		log_u = jnp.log(jax.random.uniform(subkey, shape=(i.shape[0],)))
		log_alpha = (self.temps[i] - self.temps[j]) * (ll[j] - ll[i])
		accept = log_u < log_alpha

		perm = jnp.arange(self.num_temps)
		pi = perm[i]
		pj = perm[j]

		perm = perm.at[i].set(jnp.where(accept, pj, pi))
		perm = perm.at[j].set(jnp.where(accept, pi, pj))

		new_z = z.reshape(self.num_temps, x.shape[1], *z.shape[1:])[perm]
		return new_z.reshape(self.num_temps * x.shape[1], *z.shape[1:])

	def sample_posterior(self, key, x):
		x = jnp.expand_dims(x, 0)

		z0, key = self.nuts_init(key, x.shape[1] * self.num_temps)

		def log_powerpost(z: jax.Array) -> jax.Array:
			ll = self.expanded_ll(x, z).mean(axis=1)
			return (self.temps * ll).sum() + self.ebm.logprior(z)

		def xchange(key_i: jax.Array, z_i: jax.Array, idx: jax.Array) -> jax.Array:
			return self.replica_xchange(key_i, z_i, idx, x)

		z_post = self.posterior_sampler(key, log_powerpost, z0, xchange)

		self.adapt_temps(x, z_post)

		return z_post.reshape(
			self.num_temps,
			x.shape[1],
			*z0.shape[1:],
		)

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""
		Thermodynamic integration with trapezoidal rule

		1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
		"""
		x = jnp.expand_dims(x, 0)
		delta_t = self.temps[1:] - self.temps[:-1]
		expectations = self.expanded_ll(x, z_post).mean(axis=1)
		trapz = delta_t * (expectations[1:] + expectations[:-1])
		return 0.5 * trapz.sum()
