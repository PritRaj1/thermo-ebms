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

	def get_value(v):
		return v[...]

	values_before = jax.tree.map(get_value, state_before)
	values_after = jax.tree.map(get_value, state_after)

	identical_per_node = jax.tree.map(
		lambda p1, p2: jnp.allclose(p1, p2, atol=1e-7, rtol=1e-5),
		values_before,
		values_after,
	)

	diffs = jax.tree.map(
		lambda p1, p2: jnp.sum(jnp.abs(p1 - p2)), values_before, values_after
	)
	total_change = jax.tree.reduce(lambda acc, x: acc + x, diffs, 0.0)

	print("Parameters identical per node:")
	print(identical_per_node)
	print(f"Total param change (sum |Δ|): {float(total_change):.4e}")
	print(f"Loss improvement: {float(loss_before - loss_after):.4e}")
	print(f"Grad norm: {grad_norm}")

	assert float(total_change) > 1e-8, (
		f"Parameters did not update: (change = {float(total_change):.2e})"
	)
	assert jnp.isfinite(loss_after), "Loss is NaN"


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
