import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config

cfg = make_config()


def params_change(trainer, x):
	key = jax.random.key(0)

	params_before = nnx.state(trainer.st.model, nnx.Param)
	loss, new_key = trainer.train_step(x, 1, key)
	params_after = nnx.state(trainer.st.model, nnx.Param)

	diffs = jax.tree.map(
		lambda a, b: jnp.max(jnp.abs(a - b)), params_before, params_after
	)
	total_max_change = float(
		jax.tree_util.tree_reduce(jnp.maximum, diffs, initializer=0.0)
	)

	assert total_max_change > 1e-8, (
		f"Parameters did not change. Max change = {total_max_change:.2e}"
	)
	return loss, new_key


def test_mle():
	cfg.model.thermo.num_temps = -1
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	params_change(trainer, batch["x"])


def test_thermo():
	cfg.model.thermo.num_temps = 10
	trainer = ebmTrainer(cfg)
	batch = next(iter(trainer.train_loader))
	params_change(trainer, batch["x"])
