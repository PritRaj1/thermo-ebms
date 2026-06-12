from clu import metric_writers
from clu import periodic_actions

import orbax.checkpoint as ocp
from flax import nnx
from .loaders import get_loaders
from ..networks import mleEBM, thermoEBM


def load_config(config_path: str) -> ml_collections.ConfigDict:
	"""Loads YAML configuration into a ConfigDict."""
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return ml_collections.ConfigDict(config)


class ebmTrainer:
	def __init__(
		self,
		config_path: str,
		rngs: nnx.Rngs,
	):
		config = load_config(config_path, rngs)
		model = (
			thermoEBM(config, rngs)
			if self.config.thermo.num_temps > 1
			else mleEBM(config, rngs)
		)

		opt = coupled_opt(config)
		self.st = nnx.ModelAndOptimizer(model, opt)
		nnx.display(self.st.model)

		self.writer = metric_writers.create_default_writer(logdir=config.logging.logdir)

		self.progress = periodic_actions.ReportProgress(
			num_train_steps=model.num_updates
		)

		self.ckpt_manager = ocp.CheckpointManager(
			config.logging.ckpt_dir,
			options=ocp.CheckpointManagerOptions(
				max_to_keep=5,
				save_interval_steps=1000,
				create=True,
			),
		)
