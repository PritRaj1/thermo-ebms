import jax
import jax.numpy as jnp
from flax import nnx

from networks import latentEBM
from utils import make_config


def test_forward_generates_samples():
	cfg = make_config()
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	model = latentEBM(cfg, rngs)

	N = 4
	x = model(key, N)
	assert x.shape[0] == N
	assert x.shape[1:] == (32, 32, 3)
