import jax
import jax.numpy as jnp
from flax import nnx
from utils import make_config, make_x

from thermo_ebms import KAEM, ImportanceTuner, mleEBM

cfg = make_config(z_dim=8)
x = make_x(batch=5)


def test_mle_finite():
	key = jax.random.key(0)
	model = mleEBM(cfg.model, nnx.Rngs(key))
	tuner = ImportanceTuner("residual")

	loss = tuner(key, model, x)
	assert jnp.isfinite(loss)


def test_kaem_finite():
	key = jax.random.key(0)
	model = KAEM(cfg.model, nnx.Rngs(key))
	tuner = ImportanceTuner("residual")

	loss = tuner(key, model, x)
	assert jnp.isfinite(loss)
