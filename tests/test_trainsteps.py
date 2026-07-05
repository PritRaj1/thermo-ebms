import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config

cfg = make_config()


def ps_change(trainer, x):
	key = jax.random.key(0)

	ps_before = nnx.state(trainer.model, nnx.Param)
	loss, grad_norm, new_key = trainer.train_step(x, 1, key)

	assert jnp.isfinite(grad_norm)

	ps_after = nnx.state(trainer.model, nnx.Param)
	diffs = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), ps_before, ps_after)
	total_max_change = jax.tree.reduce(jnp.maximum, diffs, 0.0)
	assert float(total_max_change) > 1e-8, (
		f"Parameters did not change. Max change = {total_max_change:.2e}, grad_norm = {float(grad_norm)}"
	)
	return loss, new_key


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
