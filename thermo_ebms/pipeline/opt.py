import jax
import optax
import flax.nnx as nnx
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


def is_leaf(x):
	return isinstance(x, nnx.VariableState)


def label_fn(path, _):
	return path[0].key


def coupled_opt(
	model: nnx.Module,
	config: ModelConfig,
	schedule: ScheduleConfig,
	updates_per_epoch: int,
):
	step = schedule.epoch_step * updates_per_epoch
	begin = schedule.epoch_begin * updates_per_epoch

	state = nnx.state(model, nnx.Param)
	params = jax.tree_util.tree_map(lambda v: v.value, state, is_leaf=is_leaf)
	labels = jax.tree_util.tree_map_with_path(label_fn, params, is_leaf=is_leaf)

	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm.optim, step, begin),
			"gen": network_opt(config.gen.optim, step, begin),
		},
		labels,
	)
