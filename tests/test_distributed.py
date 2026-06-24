import os
import sys
import jax
import pytest
import orbax.checkpoint as ocp
from absl import flags

from utils import make_config


@pytest.fixture(scope="function")
def virtual_cluster():
	"""Emulate 4 devices"""
	original_flags = os.environ.get("XLA_FLAGS", "")
	os.environ["XLA_FLAGS"] = (
		f"{original_flags} --xla_force_host_platform_device_count=4".strip()
	)

	jax.config.update("jax_platforms", "cpu")
	yield

	if original_flags:
		os.environ["XLA_FLAGS"] = original_flags
	else:
		del os.environ["XLA_FLAGS"]


@pytest.fixture
def run_slow(request):
	if not request.config.getoption("--run-slow"):
		pytest.skip("need --run-slow option to run")


def test_run_multinode(tmp_path, virtual_cluster, run_slow):
	assert jax.device_count() == 4, "4-node cluster emulation failed."

	# parse flags to satisfy grain_enable_multiprocess_worker_profiling
	if not flags.FLAGS.is_parsed():
		flags.FLAGS(sys.argv, known_only=True)

	cfg = make_config()
	key = jax.random.key(0)

	cfg.training.global_batch_size = 4
	cfg.logging.logdir = str(tmp_path / "logs")
	cfg.logging.ckpt_dir = str(tmp_path / "ckpt")

	from thermo_ebms.pipeline import ebmTrainer

	# Run cpu
	with jax.disable_jit():
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
