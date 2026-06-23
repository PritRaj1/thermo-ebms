import jax
import hydra
from flax import nnx

from thermo_ebms.pipeline import ebmTrainer


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
	key = jax.random.PRNGKey(cfg.model.seed)
	trainer = ebmTrainer(cfg, rngs=nnx.Rngs(key))
	key = trainer.run(key)


if __name__ == "__main__":
	main()
