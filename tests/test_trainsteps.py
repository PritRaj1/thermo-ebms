import jax
import jax.numpy as jnp
from flax import nnx
from utils import make_config

from thermo_ebms.pipeline import ebmTrainer

cfg = make_config()


def ps_change(trainer, x):
	key = jax.random.key(0)

	state_before = nnx.state(trainer.st.model, nnx.Param)
	values_before = jax.tree.map(lambda v: jnp.array(v[...]), state_before)
	loss_before = trainer.train_step(x, 1, key)[0]

	loss_after, grad_norm, _ = trainer.train_step(x, 1, key)

	state_after = nnx.state(trainer.st.model, nnx.Param)
	values_after = jax.tree.map(lambda v: jnp.array(v[...]), state_after)

	identical_per_node = jax.tree.map(
		lambda p1, p2: jnp.allclose(p1, p2, atol=1e-7, rtol=1e-5),
		values_before,
		values_after,
	)

	any_identical = jax.tree.reduce(
		lambda acc, x: jnp.logical_or(acc, x), identical_per_node, False
	)

	diffs = jax.tree.map(
		lambda p1, p2: jnp.sum(jnp.abs(p1 - p2)), values_before, values_after
	)
	total_change = jax.tree.reduce(lambda acc, x: acc + x, diffs, 0.0)

	print("Parameters identical per node:")
	print(identical_per_node)
	print(f"Any parameter completely unchanged? {bool(any_identical)}")
	print(f"Total param change (sum |Δ|): {float(total_change):.4e}")
	print(f"Loss improvement: {float(loss_before - loss_after):.4e}")
	print(f"Grad norm: {grad_norm}")

	assert not bool(any_identical), "Some parameters did not update at all"

	assert float(total_change) > 1e-8, (
		f"Parameters did not update: (change = {float(total_change):.2e})"
	)

	assert jnp.isfinite(loss_after), "Loss is NaN or Inf"


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
