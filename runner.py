import subprocess

MODELS = ["mle_ebm", "thermo_ebm", "mle_kaem", "thermo_kaem"]
DATASETS = ["cifar10", "svhn", "celeba"]


def run(model, dataset):
	cmd = [
		"python",
		"main.py",
		f"model={model}",
		f"training={dataset}",
	]
	print("Running:", " ".join(cmd))
	subprocess.run(cmd, check=True)


def main():
	for m in MODELS:
		for d in DATASETS:
			run(m, d)


if __name__ == "__main__":
	main()
