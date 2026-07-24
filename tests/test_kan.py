import jax
from flax import nnx
import jax.numpy as jnp

from thermo_ebms.models import wavKAN
from utils import make_config

config = make_config()
P = config.model.z_dim


def test_shape():
	key = jax.random.key(0)
	config.model.kaem.mixture = False
	model = wavKAN(config.model.kaem, P, rngs=nnx.Rngs(key))

	Q = (P - 1) // 2 if config.model.kaem.mixture else 2 * P + 1
	x = jnp.ones((10, 1, Q, P))
	y = model(x)

	assert y.shape == (10, 1, Q, P)


def test_shape_mixture():
	key = jax.random.key(0)
	config.model.kaem.mixture = True
	model = wavKAN(config.model.kaem, P, rngs=nnx.Rngs(key))
	model.sample_mixture(key, 10)

	Q = (P - 1) // 2 if config.model.kaem.mixture else 2 * P + 1
	x = jnp.ones((10, 1, Q, P))
	y = model.componentwise_pdf(x)

	assert y.shape == (10, 1, 1, P)


def test_grads():
	key = jax.random.key(0)
	model = wavKAN(config.model.kaem, P, rngs=nnx.Rngs(key))
	Q = (P - 1) // 2 if config.model.kaem.mixture else 2 * P + 1
	x = jnp.ones((10, 1, Q, P))
	z = jnp.zeros((10, 1, Q, P))

	def loss_fn(x):
		y = model.loss(x, z)
		return jnp.mean(y)

	grads = jax.grad(loss_fn)(x)
	assert grads.shape == x.shape
