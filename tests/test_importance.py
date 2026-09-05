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

	z_prior = model.sample_prior(key, x.shape[0])
	subkeys = jax.random.split(key, x.shape[0])
	idx = tuner.batch_resample(subkeys, model, z_prior, x)
	z_post = z_prior[idx].reshape(-1, *z_prior.shape[1:])
	x_new = jnp.repeat(x, x.shape[0], axis=0)

	loss = tuner(model, x_new, z_post, z_prior)
	assert jnp.isfinite(loss)
	assert z_post.shape == (x.shape[0] ** 2, 1, 1, cfg.model.z_dim)


def test_kaem_finite():
	key = jax.random.key(0)
	model = KAEM(cfg.model, nnx.Rngs(key))
	tuner = ImportanceTuner("residual")
	inner_dim = 1 if cfg.model.kaem.mixture else model.ebm.Q

	z_prior = model.sample_prior(key, x.shape[0])
	subkeys = jax.random.split(key, x.shape[0])
	idx = tuner.batch_resample(subkeys, model, z_prior, x)
	z_post = z_prior[idx].reshape(-1, *z_prior.shape[1:])
	x_new = jnp.repeat(x, x.shape[0], axis=0)

	loss = tuner(model, x_new, z_post, z_prior)
	assert jnp.isfinite(loss)
	assert z_post.shape == (x.shape[0] ** 2, 1, inner_dim, cfg.model.z_dim)
