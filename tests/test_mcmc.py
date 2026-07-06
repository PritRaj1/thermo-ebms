import os
import jax
import blackjax
from flax import nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from thermo_ebms import neuralEBM
from utils import make_config


@nnx.jit
def run_chain(model, key):
	model.eval()
	z0, key = model.mcmc_init(key, 1)
	key, runkey = jax.random.split(key)
	kernel = blackjax.sghmc(
		grad_estimator=model.ebm.prior_score,
		num_integration_steps=model.prior_sampler.L,
		alpha=model.prior_sampler.alpha,
		beta=model.prior_sampler.beta,
	)
	state = kernel.init(z0)

	def step(carry, _):
		st, newkey = carry
		newkey, subkey = jax.random.split(newkey)
		st = kernel.step(subkey, st, minibatch=None, step_size=model.prior_sampler.eta)
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
	cfg.model.ebm.mcmc.numsteps = 50
	model = neuralEBM(cfg.model, rngs)
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
