import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms import thermoEBM
from utils import make_config, make_x

cfg = make_config()
x = make_x(batch=5)


def test_shape():
	key = jax.random.key(0)
	model = thermoEBM(cfg, nnx.Rngs(key))
	model.eval()
	z = model.sample_posterior(key, x)
	assert z.shape == (cfg.thermo.num_temps, 5, 1, 1, cfg.model.z_dim)


def test_sampling():
	key = jax.random.key(0)
	model = thermoEBM(cfg, nnx.Rngs(key))
	model.eval()
	z = model.sample_posterior(key, x)

	var = jnp.var(z)
	assert jnp.isfinite(var)
	assert var > 1e-6


def test_thermo_loss():
	key = jax.random.key(0)
	model = thermoEBM(cfg, nnx.Rngs(key))
	model.eval()

	key, prior_key, post_key = jax.random.split(key, 3)
	z_post = jax.random.normal(
		post_key,
		(5 * cfg.thermo.num_temps, 1, 1, cfg.model.z_dim),
	)
	z_prior = jax.random.normal(
		prior_key,
		(5, 1, 1, cfg.model.z_dim),
	)

	loss = model.loss(x, z_post, z_prior)
	assert jnp.isfinite(loss)
