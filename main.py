import jax
import hydra
from flax import nnx
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
	cfg = apply_overrides(cfg)
	key = jax.random.PRNGKey(cfg.model.seed)
	trainer = ebmTrainer(cfg, rngs=nnx.Rngs(key))
	key = trainer.run(key)


if __name__ == "__main__":
	main()
