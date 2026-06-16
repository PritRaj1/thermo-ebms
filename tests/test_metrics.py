import yaml
import h5py
import numpy as np

from thermo_ebms.pipeline import UnbiasedMetrics
from utils import make_config


def test_metrics(tmp_path):
	cfg = make_config()
	run_dir = tmp_path / "logs"
	run_dir.mkdir()
	cfg.logging.logdir = str(run_dir)

	config_path = run_dir / "config_copy.yaml"
	h5_path = run_dir / "generated_samples.h5"
	out_json_path = run_dir / "metrics_output.json"

	with open(config_path, "w") as f:
		yaml.safe_dump(cfg.to_dict(), f)

	mock_samples = np.random.randint(0, 256, size=(2000, 32, 32, 3), dtype=np.uint8)
	with h5py.File(h5_path, "w") as f:
		f.create_dataset("samples", data=mock_samples)

	evaluator = UnbiasedMetrics(run_dir=run_dir)
	metrics = evaluator.save(out_json_path)

	assert out_json_path.exists(), "Metrics JSON file was not saved."
	assert "fid_infinity" in metrics
	assert "kid_infinity" in metrics
	assert isinstance(metrics["fid_infinity"], float)
