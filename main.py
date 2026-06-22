import jax
import hydra

from thermo_ebms.pipeline import ebmTrainer


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
	key = jax.random.PRNGKey(cfg.model.seed)
	trainer = ebmTrainer(cfg, rngs=key)
	key = trainer.run(key)


if __name__ == "__main__":
	main()
