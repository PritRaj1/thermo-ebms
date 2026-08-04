import modal

image = (
	modal.Image.debian_slim(python_version="3.12")
	.uv_sync()
	.add_local_python_source(".")
)

app = modal.App("kaem-ddp", image=image)


@app.function(
	gpu="L4:2",
	# gpu="L4:4",
	# gpu="A10:2",
	timeout=4 * 60 * 60,
	memory=65536,
)
def train(model: str, dataset: str):
	import subprocess
	import sys

	cmd = [
		"torchrun",
		"--nproc_per_node=2",
		"main.py",
		f"model={model}",
		f"training={dataset}",
	]

	print("→", " ".join(cmd))
	subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)


@app.local_entrypoint()
def main():
	models = ["mle_ebm", "mle_kaem", "thermo_ebm", "thermo_kaem"]
	datasets = ["cifar10", "svhn", "celeba"]
	list(train.starmap([(m, d) for m in models for d in datasets]))
