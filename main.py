from hydra import compose, initialize

MODELS = [
	"mle_ebm",
	"thermo_ebm",
	"mle_kaem",
	"thermo_kaem",
]

DATASETS = [
	"cifar10",
	"svhn",
	"celeba",
]


def load_config(model: str, dataset: str):
	with initialize(config_path="config", version_base=None):
		cfg = compose(
			config_name="base",
			overrides=[
				f"model={model}",
				f"dataset={dataset}",
			],
		)
	return cfg


def main():
	for model in MODELS:
		for dataset in DATASETS:
			cfg = load_config(model, dataset)

			print("=" * 80)
			print(f"{model=} {dataset=}")
			print(cfg)


if __name__ == "__main__":
	main()
