import jax
import hydra
import sys
from absl import flags
from omegaconf import OmegaConf

from thermo_ebms.pipeline import ebmTrainer


def apply_overrides(cfg):
	cfg_dict = OmegaConf.to_container(cfg, resolve=True)
	overrides = cfg_dict.get("training", {}).get("model_overrides", {})

	def deep_merge(base, override):
		for k, v in override.items():
			if isinstance(v, dict) and k in base:
				base[k] = deep_merge(base[k], v)
			else:
				base[k] = v
		return base

	cfg_dict["model"] = deep_merge(cfg_dict["model"], overrides)
	cfg_dict["training"].pop("model_overrides", None)
	return OmegaConf.create(cfg_dict)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
	# Force parse flags before any grain accesses them
	if not flags.FLAGS.is_parsed():
		flags.FLAGS(sys.argv, known_only=True)

	cfg = apply_overrides(cfg)
	key = jax.random.PRNGKey(cfg.model.seed)
	trainer = ebmTrainer(cfg)
	key = trainer.run(key)


if __name__ == "__main__":
	main()
