import os

os.environ["JAX_PLATFORMS"] = "cpu"

_original_xla_flags = os.environ.get("XLA_FLAGS", "")
_virtual_device_flag = "--xla_force_host_platform_device_count=4"
if _virtual_device_flag not in _original_xla_flags:
	os.environ["XLA_FLAGS"] = f"{_original_xla_flags} {_virtual_device_flag}".strip()

import jax
import orbax.checkpoint as ocp
import pytest
from absl import flags
from utils import make_config

from thermo_ebms.pipeline import ebmTrainer


@pytest.fixture(scope="function")
def virtual_cluster():
	"""Emulate a 4-device CPU cluster on a single host."""

	assert jax.default_backend() == "cpu", (
		f"Expected CPU backend, got {jax.default_backend()!r}"
	)

	assert jax.device_count() == 4, (
		f"Expected 4 virtual CPU devices, got {jax.device_count()}"
	)

	yield


def test_run_multinode(tmp_path, virtual_cluster):
	"""Run training job across 4 virtual JAX CPUs."""

	# Parse Abseil flags so Grain can safely access.
	if not flags.FLAGS.is_parsed():
		flags.FLAGS(["pytest"], known_only=True)

	cfg = make_config()
	key = jax.random.key(0)

	cfg.training.global_batch_size = 4
	cfg.logging.logdir = str(tmp_path / "logs")
	cfg.logging.ckpt_dir = str(tmp_path / "ckpt")

	trainer = ebmTrainer(cfg)
	trainer.run(key)

	logdir = tmp_path / "logs"
	assert logdir.exists(), "Log directory was not created."
	assert (logdir / "config_copy.yaml").exists(), "Configuration copy was not written."

	ckpt_dir = tmp_path / "ckpt"
	assert ckpt_dir.exists(), "Checkpoint directory was not created."
	assert any(ckpt_dir.iterdir()), "No checkpoints were written."

	mngr = ocp.CheckpointManager(str(ckpt_dir))

	try:
		latest_step = mngr.latest_step()
		assert latest_step is not None, "No checkpoint step was found."
		assert latest_step > 0, "Training did not advance beyond step 0."
	finally:
		mngr.close()

	h5_file = logdir / "generated_samples.h5"
	assert h5_file.exists(), "HDF5 samples file was not created."
