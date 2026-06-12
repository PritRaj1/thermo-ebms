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
		return (st, newkey), st.position

	(_, _), z = jax.lax.scan(
		step,
		(state, key),
		xs=None,
		length=model.ula_iters_prior,
	)

	return z


def test_mala_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	model = latentEBM(cfg, rngs)

	traj = run_chain(model, key)
	energy = jax.vmap(lambda z: model.ebm.logprior(z.squeeze()))(traj)

	os.makedirs("debug_plots", exist_ok=True)

	plt.figure()
	plt.plot(energy)
	plt.title("MALA prior energy")
	plt.xlabel("Step")
	plt.ylabel("Energy")
	plt.savefig("debug_plots/energy_traj.png", dpi=150)
	plt.close()

	var = jnp.var(traj.reshape(traj.shape[0], -1), axis=1)

	plt.figure()
	plt.plot(var)
	plt.title("Latent variance over time")
	plt.xlabel("Step")
	plt.ylabel("Variance")
	plt.savefig("debug_plots/variance_traj.png", dpi=150)
	plt.close()

	assert jnp.isfinite(energy).all()
	assert jnp.std(energy) > 0.0
	assert jnp.std(var) > 0.0
