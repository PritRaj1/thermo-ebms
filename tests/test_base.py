import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms import neuralEBM
from utils import make_config


def test_sample_prior():
	cfg = make_config()

	key = jax.random.key(0)
	model = neuralEBM(cfg, nnx.Rngs(key))

	N = 4
	z = model.sample_prior(key, N)

	assert z.shape == (N, 1, 1, cfg.model.z_dim)


def test_forward_generates_samples():
	cfg = make_config()
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)
	model = neuralEBM(cfg, rngs)

	N = 4
	x = model(key, N)
	assert x.shape[0] == N
	assert x.shape[1:] == (32, 32, 3)


def test_contrastive_divergence():
	cfg = make_config()
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)
	model = neuralEBM(cfg, rngs)

	N = 4
	z_prior = model.sample_prior(key, N)

	z_post = jax.random.normal(
		key,
		(N, 1, 1, cfg.model.z_dim),
	)

	loss = model.ebm.loss(z_post, z_prior)
	assert loss.shape == ()
