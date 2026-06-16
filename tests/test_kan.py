import jax
import jax.numpy as jnp

from thermo_ebms.models import kanBANK
from utils import make_config

config = make_config()
P = config.model.z_dim
Q = (P - 1) // 2 if config.kan_prior.mixture else 2 * P + 1


def test_shape():
	model = kanBANK(config, P=P, Q=Q)

	x = jnp.ones((10, P))
	y = model(x)

	assert y.shape == (10, Q, P)


def test_grads():
	model = kanBANK(config, P=P, Q=Q)
	x = jnp.ones((10, P))

	def loss_fn(x):
		y = model(x)
		return jnp.mean(y)

	grads = jax.grad(loss_fn)(x)

	assert grads.shape == (10, P)
