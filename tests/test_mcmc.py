import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms import neuralEBM
from utils import make_config


@nnx.jit
def run_chain(model, key):
	model.eval()
	z0, key = model.mcmc_init(key, 1)
	key, runkey = jax.random.split(key)

	def step(carry, _):
		z, newkey = carry
		newkey, subkey = jax.random.split(newkey)
		eps = jax.random.normal(subkey, z.shape)
		z += (
			model.prior_sampler.eta * model.ebm.prior_score(z)
			+ jnp.sqrt(2 * model.prior_sampler.eta) * eps
		)
		return (z, newkey), z

	(_, _), z = jax.lax.scan(
		step,
		(z0, runkey),
		xs=None,
		length=model.prior_sampler.run_iters,
	)

	return z


def test_mcmc_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	cfg.ebm.mcmc_numsteps = 50
	model = neuralEBM(cfg, rngs)
	z = run_chain(model, key)
	energy = jax.vmap(model.ebm.prior_score)(z)

	os.makedirs("debug_plots", exist_ok=True)

	fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 6))

	z = z.reshape(z.shape[0], -1)
	ax.plot(z[:, 0])
	ax.set_xlabel("Samples")
	ax.set_ylabel("z[0]")

	ax1.plot(jnp.linalg.norm(z, axis=-1))
	ax1.set_xlabel("Samples")
	ax1.set_ylabel("||z||")

	plt.savefig("debug_plots/mcmc_traj.png", dpi=150)
	plt.close()

	assert jnp.isfinite(energy).all()
	assert jnp.std(energy) > 0.0
