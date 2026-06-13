import optax
from ml_collections import ConfigDict
from collections.abc import Callable


def network_opt(config: ConfigDict, step: int, begin: int) -> Callable:
	schedule = optax.exponential_decay(
		init_value=config.lr_init,
		transition_steps=step,
		decay_rate=config.lr_decay,
		transition_begin=begin,
		end_value=config.lr_end,
	)

	return optax.adam(schedule, config.lr_beta1, config.lr_beta2)


def coupled_opt(config: ConfigDict, updates_per_epoch: int):
	begin = config.lr_schedule.begin * updates_per_epoch
	step = config.lr_schedule.step * updates_per_epoch
	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm, step, begin),
			"gen": network_opt(config.gen, step, begin),
		},
		{"ebm": "ebm", "gen": "gen"},
	)
