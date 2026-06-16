import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer, train_step
from utils import make_config

cfg = make_config()


def test_mle():
	key = jax.random.key(0)

	cfg.thermo.num_temps = -1
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	x = batch["x"]

	graph, params_before, state = nnx.split(trainer.model, nnx.Param, ...)
	trainer.model, trainer.opt_st, loss, key = train_step(
		trainer.tx, trainer.opt_st, trainer.model, x, key
	)
	_, params_after, _ = nnx.split(trainer.model, nnx.Param, ...)

	before_flat = jax.tree_util.tree_leaves(params_before)
	after_flat = jax.tree_util.tree_leaves(params_after)

	diffs = [jnp.max(jnp.abs(a - b)) for a, b in zip(before_flat, after_flat)]

	total_change = jnp.max(jnp.stack(diffs))
	assert jnp.isfinite(total_change)
	assert total_change > 0.0, "Parameters did not change after update"


def test_thermo():
	key = jax.random.key(0)

	cfg.thermo.num_temps = 10
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	x = batch["x"]

	graph, params_before, state = nnx.split(trainer.model, nnx.Param, ...)
	trainer.model, trainer.opt_st, loss, key = train_step(
		trainer.tx, trainer.opt_st, trainer.model, x, key
	)
	_, params_after, _ = nnx.split(trainer.model, nnx.Param, ...)

	before_flat = jax.tree_util.tree_leaves(params_before)
	after_flat = jax.tree_util.tree_leaves(params_after)

	diffs = [jnp.max(jnp.abs(a - b)) for a, b in zip(before_flat, after_flat)]

	total_change = jnp.max(jnp.stack(diffs))
	assert jnp.isfinite(total_change)
	assert total_change > 0.0, "Parameters did not change after update"
