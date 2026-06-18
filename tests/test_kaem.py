import jax
from flax import nnx

from thermo_ebms import KAEM
from utils import make_config


config = make_config(z_dim=8)


def test_sample_prior_shape():
	key = jax.random.key(0)
	model = KAEM(config, rngs=nnx.Rngs(key))

	N = 10
	z = model.sample_prior(key, N)

	inner_dim = 1 if config.kaem.mixture else model.ebm.f.Q
	assert z.shape == (N, 1, inner_dim, model.z_dim)
