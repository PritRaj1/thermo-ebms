import jax
from hydra import compose, initialize


def make_config(z_dim=8, num_temps=4):
	with initialize(config_path="../config", version_base=None):
		cfg = compose(
			config_name="test",
			overrides=[
				f"model.z_dim={z_dim}",
				f"model.thermo.num_temps={num_temps}",
			],
		)
	return cfg


def make_x(batch=4):
	return jax.random.normal(jax.random.key(0), (batch, 32, 32, 3))
