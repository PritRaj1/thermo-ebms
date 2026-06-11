import yaml
import ml_collections
from src.data.loaders import get_loaders


def load_config(config_path: str) -> ml_collections.ConfigDict:
	"""Loads YAML configuration into a ConfigDict."""
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return ml_collections.ConfigDict(config)


def main():
	config = load_config("config/conf.yaml")
	train_loader, test_loader = get_loaders(config.dataset)
	train_iter = iter(train_loader)
	print(train_iter)
	sample = next(train_iter)


if __name__ == "__main__":
	main()
