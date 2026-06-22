import jax
import flax.nnx as nnx
import optax
from collections.abc import Callable

from ..config import ModelConfig, OptConfig, ScheduleConfig


def network_opt(config: OptConfig, step: int, begin: int) -> Callable:
	schedule = optax.exponential_decay(
		init_value=config.lr_init,
		transition_steps=step,
		decay_rate=config.lr_decay,
		transition_begin=begin,
		end_value=config.lr_end,
	)

	return optax.adam(schedule, config.beta1, config.beta2)


def get_partition(path, leaf):
	return path[0].key


def coupled_opt(
	model: nnx.Module,
	config: ModelConfig,
	schedule: ScheduleConfig,
	updates_per_epoch: int,
):
	st = nnx.state(model, nnx.Param)
	step = schedule.epoch_step * updates_per_epoch
	begin = schedule.epoch_begin * updates_per_epoch

	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm.optim, step, begin),
			"gen": network_opt(config.gen.optim, step, begin),
		},
		jax.tree.map_with_path(get_partition, st),
	)
