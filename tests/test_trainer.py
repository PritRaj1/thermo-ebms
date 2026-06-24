import jax
import sys
from absl import flags
import orbax.checkpoint as ocp

from thermo_ebms.pipeline import ebmTrainer
from utils import make_config


def test_logdir(tmp_path):

	# parse flags to satisfy grain_enable_multiprocess_worker_profiling
	if not flags.FLAGS.is_parsed():
		flags.FLAGS(sys.argv, known_only=True)

	cfg = make_config()
	key = jax.random.key(0)

	cfg.logging.logdir = str(tmp_path / "logs")
	cfg.logging.ckpt_dir = str(tmp_path / "ckpt")

	trainer = ebmTrainer(cfg)
	trainer.run(key)

	logdir = tmp_path / "logs"
	assert logdir.exists()
	assert (logdir / "config_copy.yaml").exists()

	ckpt_dir = tmp_path / "ckpt"
	assert ckpt_dir.exists()
	assert any(ckpt_dir.iterdir()), "No checkpoints were written"

	h5_file = logdir / "generated_samples.h5"
	assert h5_file.exists(), "HDF5 samples file not created"

	mngr = ocp.CheckpointManager(str(ckpt_dir))
	latest_step = mngr.latest_step()
	assert latest_step is not None
	assert latest_step > 0
