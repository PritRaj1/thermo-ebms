from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from tempfile import TemporaryDirectory
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch_fidelity import calculate_metrics


class UnbiasedMetrics:
	def __init__(
		self,
		run_dir: str | Path,
		real_samples_path: str | Path,
	):
		self.run_dir = Path(run_dir)
		self.real_samples_path = Path(real_samples_path)
		self.config = self._load_config()

		metrics_cfg = self.config["unbiased_metrics"]

		self.batch_sizes = metrics_cfg["batch_sizes"]
		self.seed = int(metrics_cfg.get("seed", 0))
		self.rng = np.random.default_rng(self.seed)

		self.generated_file = self.run_dir / "generated_samples.h5"
		if not self.generated_file.exists():
			raise FileNotFoundError(self.generated_file)

	def _load_config(self) -> dict:
		cfg_path = self.run_dir / "config_copy.yaml"
		if not cfg_path.exists():
			raise FileNotFoundError(cfg_path)

		with open(cfg_path) as f:
			return yaml.safe_load(f)

	@staticmethod
	def _load_h5(path: Path) -> np.ndarray:
		with h5py.File(path, "r") as f:
			return np.asarray(f["samples"])

	def _fid_kid(
		self,
		real_dir: str,
		gen_dir: str,
	) -> dict:
		return calculate_metrics(
			input1=real_dir,
			input2=gen_dir,
			fid=True,
			kid=True,
			isc=False,
			verbose=False,
		)

	def _fit_infinity(
		self,
		batch_sizes: list[int],
		values: list[float],
	):
		x = (1.0 / np.asarray(batch_sizes)).reshape(-1, 1)
		y = np.asarray(values).reshape(-1, 1)

		reg = LinearRegression().fit(x, y)
		intercept = float(reg.predict(np.array([[0.0]]))[0, 0])

		n = len(batch_sizes)
		x_flat = x.flatten()
		x_mean = x_flat.mean()
		ss_x = np.sum((x_flat - x_mean) ** 2)

		pred = reg.predict(x).flatten()
		residuals = y.flatten() - pred
		mse = np.sum(residuals**2) / (n - 2)

		std_err = float(np.sqrt(mse * (1 / n + x_mean**2 / ss_x)))
		r2 = float(reg.score(x, y))
		return intercept, std_err, r2

	def evaluate(self) -> dict:
		generated = self._load_h5(self.generated_file)
		real = self._load_h5(self.real_samples_path)

		n_gen = len(generated)
		batch_sizes = [b for b in self.batch_sizes if b <= n_gen]

		if len(batch_sizes) < 3:
			raise ValueError(
				f"Need at least 3 valid batch sizes. "
				f"Got {len(batch_sizes)} usable sizes."
			)

		fids = []
		kids = []

		def write_images(arr, directory):
			directory = Path(directory)
			directory.mkdir(parents=True, exist_ok=True)
			for i, img in enumerate(arr):
				Image.fromarray(img).save(directory / f"{i:06d}.png")

		with TemporaryDirectory() as real_dir:
			real_dir = Path(real_dir)
			write_images(real, real_dir)

			for b in batch_sizes:
				idx = self.rng.choice(n_gen, size=b, replace=False)
				subset = generated[idx]

				with TemporaryDirectory() as gen_dir:
					gen_dir = Path(gen_dir)
					write_images(subset, gen_dir)

					metrics = self._fid_kid(
						str(real_dir),
						str(gen_dir),
					)

				fids.append(float(metrics["frechet_inception_distance"]))
				kids.append(float(metrics["kernel_inception_distance_mean"]))

		fid_inf, fid_se, fid_r2 = self._fit_infinity(batch_sizes, fids)
		kid_inf, kid_se, kid_r2 = self._fit_infinity(batch_sizes, kids)
		return {
			"run_dir": str(self.run_dir),
			"real_samples": str(self.real_samples_path),
			"num_generated": int(n_gen),
			"num_real": int(len(real)),
			"batch_sizes": batch_sizes,
			"fid_values": fids,
			"kid_values": kids,
			"fid_infinity": fid_inf,
			"fid_std_error": fid_se,
			"fid_r2": fid_r2,
			"kid_infinity": kid_inf,
			"kid_std_error": kid_se,
			"kid_r2": kid_r2,
		}

	def save(self, out_path: str | Path) -> dict:
		out_path = Path(out_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		metrics = self.evaluate()

		with open(out_path, "w") as f:
			json.dump(metrics, f, indent=2)

		return metrics
