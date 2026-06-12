import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from networks import thermoEBM
from utils import make_config, make_x


def test_shape():
	cfg = make_config()
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	model = thermoEBM(cfg, rngs)
	x = make_x(batch=4)
	z = model.sample_posterior(key, x)

	assert z.shape == (cfg.thermo.num_temps - 1, 4, 1, 1, cfg.model.z_dim)


def test_sampling():
	cfg = make_config()
	key = jax.random.key(0)

	model = thermoEBM(cfg, nnx.Rngs(key))
	x = make_x()
	z = model.sample_posterior(key, x)

	var = jnp.var(z)
	assert jnp.isfinite(var)
	assert var > 1e-6
