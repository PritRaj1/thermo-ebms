import jax
import jax.numpy as jnp
import pytest
from ml_collections import ConfigDict
from flax import nnx

from networks import mleEBM


def make_config(z_dim=8, num_temps=5):
	cfg = ConfigDict()

	cfg.model = ConfigDict()
	cfg.model.z_dim = z_dim

	cfg.ebm = ConfigDict()
	cfg.ebm.energy_dim = 32
	cfg.ebm.leakyrelu_leak = 0.1
	cfg.ebm.p0_stddev = 1.0
	cfg.ebm.ula_eta = 1e-3
	cfg.ebm.ula_numsteps = 100

	cfg.gen = ConfigDict()
	cfg.gen.hidden_dim = 32
	cfg.gen.img_channels = 3
	cfg.gen.image_res = 32
	cfg.gen.leakyrelu_leak = 0.1
	cfg.gen.gaussian_stddev = 1.0
	cfg.gen.ula_eta = 1e-3
	cfg.gen.ula_numsteps = 100

	cfg.thermo = ConfigDict()
	cfg.thermo.num_temps = num_temps
	cfg.thermo.annealing_cycles = 1.0
	cfg.thermo.p_start = 0.5
	cfg.thermo.p_end = 2.0

	cfg.training = ConfigDict()
	cfg.training.epochs = 100

	return cfg


def make_x(batch=4):
	return jax.random.normal(jax.random.key(0), (batch, 32, 32, 3))


def test_shape():
	cfg = make_config()
	key = jax.random.key(0)

	model = mleEBM(cfg, nnx.Rngs(key))
	x = make_x(batch=4)
	z = model.sample_posterior(key, x)

	assert z.shape == (4, 1, 1, cfg.model.z_dim)


def test_sampling():
	cfg = make_config()
	key = jax.random.key(0)

	model = mleEBM(cfg, nnx.Rngs(key))
	x = make_x()
	z = model.sample_posterior(key, x)

	var = jnp.var(z)
	assert jnp.isfinite(var)
	assert var > 1e-6
