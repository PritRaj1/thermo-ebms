import jax
import flax.nnx as nnx
import optax
from collections.abc import Callable

from ..config import ModelConfig, OptConfig


def network_opt(config: OptConfig, updates_per_epoch: int) -> Callable:
	schedule = optax.exponential_decay(
		init_value=config.lr_init,
		transition_steps=config.epoch_step * updates_per_epoch,
		decay_rate=config.lr_decay,
		transition_begin=config.epoch_begin * updates_per_epoch,
		end_value=config.lr_end,
	)

	return optax.adam(schedule, config.beta1, config.beta2)


def get_partition(path, leaf):
	return path[0].key


def coupled_opt(model: nnx.Module, config: ModelConfig, updates_per_epoch: int):
	st = nnx.state(model, nnx.Param)

	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm.optim, updates_per_epoch),
			"gen": network_opt(config.gen.optim, updates_per_epoch),
		},
		jax.tree.map_with_path(get_partition, st),
	)
