import typer
from pathlib import Path

from thermo_ebms.pipeline import UnbiasedMetrics

app = typer.Typer()


@app.command()
def run(run_dir: str):
	run_dir = Path(run_dir)
	m = UnbiasedMetrics(run_dir)
	out = run_dir / "metrics.json"
	m.save(out)
	print(f"Saved: {out}")


@app.command()
def all_runs(root: str):
	root = Path(root)

	for run_dir in root.iterdir():
		if (run_dir / "generated_samples.h5").exists():
			m = UnbiasedMetrics(run_dir)
			m.save(run_dir / "metrics.json")
			print(f"Finished {run_dir.name}")


if __name__ == "__main__":
	app()
