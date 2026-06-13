import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config

cfg = make_config()


def test_mle():
	key = jax.random.key(0)

	cfg.thermo.num_temps = -1
	trainer = ebmTrainer(cfg, nnx.Rngs(key))
	batch = next(iter(trainer.train_loader))
	x = batch["x"]

	graph, params_before, state = nnx.split(trainer.st.model, nnx.Param, ...)
	loss, key = trainer.update(key, x)
	_, params_after, _ = nnx.split(trainer.st.model, nnx.Param, ...)

	before_flat = jax.tree_util.tree_leaves(params_before)
	after_flat = jax.tree_util.tree_leaves(params_after)

	diffs = [jnp.max(jnp.abs(a - b)) for a, b in zip(before_flat, after_flat)]

	total_change = jnp.max(jnp.stack(diffs))
	assert jnp.isfinite(total_change)
	assert total_change > 0.0, "Parameters did not change after update"
