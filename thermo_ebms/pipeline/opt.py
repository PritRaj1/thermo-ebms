from collections.abc import Callable

import optax

from ..config import AdamConfig, OptConfig


def network_opt(config: AdamConfig, updates_per_epoch: int) -> Callable:
	step = config.decay_step * updates_per_epoch
	begin = config.decay_begin * updates_per_epoch

	schedule = optax.exponential_decay(
		init_value=config.lr_init,
		transition_steps=step,
		decay_rate=config.lr_decay,
		transition_begin=begin,
		end_value=config.lr_end,
	)

	return optax.adam(schedule, config.beta1, config.beta2)


def label_fn(path: tuple) -> str:
	path_str = str(path).lower()
	if "ebm" in path_str:
		return "ebm"
	if "kan" in path_str:
		return "kan"
	return "gen"


def coupled_opt(
	config: OptConfig,
	updates_per_epoch: int,
):

	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm, updates_per_epoch),
			"gen": network_opt(config.gen, updates_per_epoch),
			"kan": network_opt(config.kaem, updates_per_epoch),
		},
		label_fn,
	)
