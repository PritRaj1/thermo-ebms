import os
import pytest
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx
import blackjax

from networks import latentEBM
from utils import make_config


@nnx.jit
def run_chain(model, key):
	model.eval()
	z0, key = model.ula_init(key, 1)
	kernel = blackjax.mala(model.ebm.logprior, model.ula_step_prior)
	state = kernel.init(z0)

	def step(carry, _):
		st, newkey = carry
		newkey, subkey = jax.random.split(newkey)
		st, _ = kernel.step(subkey, st)
		return (st, newkey), st

	(_, _), state = jax.lax.scan(
		step,
		(state, key),
		xs=None,
		length=model.ula_iters_prior,
	)

	return state


def test_mala_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	model = latentEBM(cfg, rngs)
	traj = run_chain(model, key)
	z = traj.position
	energy = jax.vmap(lambda zi: model.ebm.logprior(zi))(z)

	os.makedirs("debug_plots", exist_ok=True)

	fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 6))

	z = z.reshape(z.shape[0], -1)
	ax.plot(z[:, 0])
	ax.set_xlabel("Samples")
	ax.set_ylabel("z[0]")

	ax1.plot(jnp.linalg.norm(z, axis=-1))
	ax1.set_xlabel("Samples")
	ax1.set_ylabel("||z||")

	plt.savefig("debug_plots/mala_traj.png", dpi=150)
	plt.close()

	assert jnp.isfinite(energy).all()
	assert jnp.std(energy) > 0.0
