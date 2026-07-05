import jax
import jax.numpy as jnp
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config

cfg = make_config()


def ps_change(trainer, x):
	key = jax.random.key(0)
	param_node = trainer.model.gen.g.layers[0].kernel

	val_before = jnp.array(param_node.value)
	loss, grad_norm, _ = trainer.train_step(x, 1, key)
	val_after = jnp.array(param_node.value)

	change = jnp.sum(jnp.abs(val_before - val_after))
	print(f"Param Change: {float(change):.4e}")

	assert float(change) > 1e-12, "Parameters did not update."


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
