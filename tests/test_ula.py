import os
import jax
import jax.numpy as jnp
from flax import nnx
import pytest
import matplotlib.pyplot as plt
from ml_collections import ConfigDict

from networks import latentEBM
from utils import make_config


def run_chain(model, key):
	z0, key = model.ula_init(key, 1)

	def step(carry, _):
		return model.ula_prior_step(carry, None)

	(zT, _), traj = jax.lax.scan(
		step,
		(z0, key),
		xs=None,
		length=model.ula_iters_prior,
	)

	return traj


@pytest.mark.skipif(
	os.getenv("SAVE_PLOTS") != "1",
	reason="SAVE_PLOTS=1 to generate plots",
)
def test_lang_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	model = latentEBM(cfg, rngs)

	traj = run_chain(model, key)

	energy = jax.vmap(model.ebm.logprior)(traj)

	os.makedirs("debug_plots", exist_ok=True)

	plt.figure()
	plt.plot(energy)
	plt.title("Langevin prior energy")
	plt.xlabel("Step")
	plt.ylabel("Energy")
	plt.savefig("debug_plots/energy_traj.png", dpi=150)
	plt.close()

	var = jnp.var(traj, axis=(1, 2, 3, 4))

	plt.figure()
	plt.plot(var)
	plt.title("Latent variance over time")
	plt.xlabel("Step")
	plt.ylabel("Variance")
	plt.savefig("debug_plots/variance_traj.png", dpi=150)
	plt.close()

	assert True


@pytest.mark.skipif(
	os.getenv("SAVE_PLOT") != "1",
	reason="SAVE_PLOT=1 to generate plots",
)
def test_lang_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	model = latentEBM(cfg, rngs)

	traj = run_chain(model, key)
	energy = jax.vmap(model.ebm.logprior)(traj)

	os.makedirs("debug_plots", exist_ok=True)

	plt.figure()
	plt.plot(energy)
	plt.title("Langevin prior energy")
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
