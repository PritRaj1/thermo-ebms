import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict

from .base import neuralEBM


def build_pairs(T, offset):
	idx = jnp.arange(offset, T, 2)
	return jnp.stack([idx, idx + 1], axis=1)


class thermoEBM(neuralEBM):
	def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
		super().__init__(config, rngs)
		self.num_temps = config.thermo.num_temps
		self.temps = jnp.linspace(0.0, 1.0, self.num_temps)

		# DEO exchange
		self.i_pairs = build_pairs(self.num_temps, 0)
		self.j_pairs = build_pairs(self.num_temps, 1)

	def adapt_temps(self, x: jax.Array, z: jax.Array) -> None:
		"""
		Adapt temps by minimising/equalizing KL div between adjacent power posteriors

		KL(p_t || p_{t+Δt}) = 0.5 Var_t[ log p_β(x | z) * Δt^2]
		"""

		def wrapped_ll(z_t: jax.Array) -> jax.Array:
			return jnp.sum((x - self.gen(z_t)) ** 2, axis=(1, 2, 3))

		ll = jax.vmap(wrapped_ll)(z)
		rho = ll.std(axis=1)
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
		def wrapped_ll(z_t: jax.Array) -> jax.Array:
			return self.gen.loss(x, z_t)

		ll = jax.vmap(wrapped_ll)(z)
		phase = step_idx % 2
		i = self.i_pairs[:, phase]
		j = self.j_pairs[:, phase]

		key, subkey = jax.random.split(key)
		log_u = jnp.log(jax.random.uniform(subkey, shape=(i.shape[0],)))
		log_alpha = (self.temps[i] - self.temps[j]) * (ll[j] - ll[i])
		accept = log_u < log_alpha

		perm = jnp.arange(self.num_temps)
		pi = perm[i]
		pj = perm[j]

		perm = perm.at[i].set(jnp.where(accept, pj, pi))
		perm = perm.at[j].set(jnp.where(accept, pi, pj))
		return z[perm]

	def sample_posterior(self, key, x):
		z0, key = self.mcmc_init(key, x.shape[0] * self.num_temps)
		z0 = z0.reshape(self.num_temps, x.shape[0], *z0.shape[1:])
		t = self.temps[:, None, None, None, None]

		def wrapped_gradll(z: jax.Array) -> jax.Array:
			return self.gen.posterior_score(z, x)

		def score(z: jax.Array) -> jax.Array:
			return (t * jax.vmap(wrapped_gradll)(z)).sum() + self.ebm.prior_score(z)

		def xchange(key_i: jax.Array, z_i: jax.Array, idx: jax.Array) -> jax.Array:
			return self.replica_xchange(key_i, z_i, idx, x)

		z0 = self.posterior_sampler(key, score, z0, xchange)
		self.adapt_temps(x, z0)
		return z0

	def loss(self, x: jax.Array, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
		"""
		Thermodynamic integration with trapezoidal rule

		1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
		"""

		def wrapped_ll(z_t: jax.Array) -> jax.Array:
			return self.gen.loss(x, z_t)

		expectations = jax.vmap(wrapped_ll)(z_post)
		delta_t = self.temps[1:] - self.temps[:-1]
		trapz = delta_t * (expectations[1:] + expectations[:-1])
		return 0.5 * trapz.sum()
