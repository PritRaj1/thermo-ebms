import jax
from flax import nnx

from thermo_ebms import neuralEBM
from utils import make_config


cfg = make_config()


def test_sample_prior():
	key = jax.random.key(0)
	model = neuralEBM(cfg, nnx.Rngs(key))
	model.eval()

	N = 4
	z = model.sample_prior(key, N)

	assert z.shape == (N, 1, 1, cfg.model.z_dim)


def test_gen():
	key = jax.random.key(0)
	model = neuralEBM(cfg, nnx.Rngs(key))
	model.eval()

	N = 4
	x, key = model(key, N)
	assert x.shape[0] == N
	assert x.shape[1:] == (32, 32, 3)


def test_contrastive_divergence():
	key = jax.random.key(0)
	model = neuralEBM(cfg, nnx.Rngs(key))
	model.eval()

	N = 4
	z_prior = model.sample_prior(key, N)

	z_post = jax.random.normal(
		key,
		(N, 1, 1, cfg.model.z_dim),
	)

	loss = model.ebm.loss(z_post, z_prior)
	assert loss.shape == ()
