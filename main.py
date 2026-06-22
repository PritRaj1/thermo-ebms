from hydra import compose, initialize


def load_config(name: str):
	with initialize(config_path="config", version_base=None):
		cfg = compose(
			config_name=name,
		)
	return cfg


def main():
	config = load_config("cifar10")
	print(config)


if __name__ == "__main__":
	main()
