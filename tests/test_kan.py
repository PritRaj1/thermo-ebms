import jax
import jax.numpy as jnp

from thermo_ebms.models import kanBANK
from utils import make_config

config = make_config()
P = config.model.z_dim
Q = (P - 1) // 2 if config.kaem.mixture else 2 * P + 1


def test_shape():
	model = kanBANK(config.kaem, P, 0)

	x = jnp.ones((10, P)) if config.kaem.mixture else jnp.ones((10, Q, P))
	y = model(x)

	assert y.shape == (10, Q, P)


def test_grads():
	model = kanBANK(config.kaem, P, 0)
	x = jnp.ones((10, P)) if config.kaem.mixture else jnp.ones((10, Q, P))

	def loss_fn(x):
		y = model(x)
		return jnp.mean(y)

	grads = jax.grad(loss_fn)(x)
	assert grads.shape == x.shape
