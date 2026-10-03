import subprocess

# MODELS = ["mle_ebm", "mle_kaem", "thermo_ebm", "thermo_kaem"]
MODELS = ["mle_kaem", "thermo_kaem"]
DATASETS = ["celeba", "cifar10", "svhn"]


def train(model, dataset):
	cmd = [
		"python",
		"main.py",
		f"model={model}",
		f"training={dataset}",
	]
	print("Training:", " ".join(cmd))
	subprocess.run(cmd, check=True)


def eval(model, dataset):
	cmd = [
		"python",
		"eval.py",
		"run",
		f"runs/{model}_{dataset}",
	]
	print("Evaluating:", " ".join(cmd))
	subprocess.run(cmd, check=True)


def main():
	for m in MODELS:
		for d in DATASETS:
			train(m, d)
			eval(m, d)


if __name__ == "__main__":
	main()
