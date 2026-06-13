import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms import mleEBM
from utils import make_config, make_x


def test_shape():
	cfg = make_config()
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	model = mleEBM(cfg, rngs)
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


def test_mle_loss():
	cfg = make_config()
	key = jax.random.key(0)

	model = mleEBM(cfg, nnx.Rngs(key))
	x = make_x(batch=4)

	key, prior_key, post_key = jax.random.split(key, 3)
	z_post = jax.random.normal(
		post_key,
		(4, 1, 1, cfg.model.z_dim),
	)
	z_prior = jax.random.normal(
		prior_key,
		(4, 1, 1, cfg.model.z_dim),
	)

	loss = model.loss(x, z_post, z_prior)
	assert jnp.isfinite(loss)
