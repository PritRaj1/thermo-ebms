import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx
import blackjax

from thermo_ebms import neuralEBM
from utils import make_config


@nnx.jit
def run_chain(model, key):
	model.eval()
	z0, key = model.mcmc_init(key, 1)
	key, runkey = jax.random.split(key)
	kernel = blackjax.mala(model.ebm.logprior, model.prior_sampler.step_size)
	state = kernel.init(z0)

	def step(carry, _):
		st, newkey = carry
		newkey, subkey = jax.random.split(newkey)
		st, _ = kernel.step(subkey, st)
		return (st, newkey), st

	(_, _), state = jax.lax.scan(
		step,
		(state, runkey),
		xs=None,
		length=model.prior_sampler.run_iters,
	)

	return state


def test_mcmc_plot():
	key = jax.random.key(0)
	rngs = nnx.Rngs(key)

	cfg = make_config()
	cfg.ebm.mcmc_burn_in = 20
	cfg.ebm.mcmc_numsteps = 1000
	model = neuralEBM(cfg, rngs)
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

	plt.savefig("debug_plots/mcmc_traj.png", dpi=150)
	plt.close()

	assert jnp.isfinite(energy).all()
	assert jnp.std(energy) > 0.0
