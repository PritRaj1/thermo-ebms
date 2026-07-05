import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config

cfg = make_config()


def ps_change(trainer, x):
	key = jax.random.key(0)
	state_before = nnx.state(trainer.st.model, nnx.Param)

	loss_before, _, _ = trainer.train_step(x, 1, key)
	loss_after, grad_norm, _ = trainer.train_step(x, 1, key)
	state_after = nnx.state(trainer.st.model, nnx.Param)

	diffs = jax.tree_util.tree_map(
		lambda a, b: jnp.sum(jnp.abs(a[...] - b[...])), state_before, state_after
	)

	total_change = jax.tree_util.tree_reduce(lambda acc, x: acc + x, diffs, 0.0)
	assert float(total_change) > 1e-6, "Not all parameters updated."
	assert jnp.isfinite(loss_after), "Loss is NaN"

	print(f"Param Change: {float(total_change):.4e}")
	print(f"Loss improvement: {float(loss_before - loss_after):.4e}")


def test_mle():
	cfg.model.thermo.num_temps = -1
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	ps_change(trainer, batch["x"])


def test_thermo():
	cfg.model.thermo.num_temps = 10
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	ps_change(trainer, batch["x"])
