import yaml
import ml_collections


def load_config(config_path: str) -> ml_collections.ConfigDict:
	"""Loads YAML configuration into a ConfigDict."""
	with open(config_path) as f:
		config = yaml.safe_load(f)
	return ml_collections.ConfigDict(config)


def main():
	config = load_config("config/conf.yaml")


if __name__ == "__main__":
	main()
