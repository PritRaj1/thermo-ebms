from dataclasses import dataclass
from collections.abc import Sequence


@dataclass
class TrainingConfig:
	dataset: str
	epochs: int = 100
	batch_size: int = 128


@dataclass
class LoggingConfig:
	logdir: str = "./logs"
	ckpt_dir: str = "./checkpoints"
	ckpt_every: int = 1000
	eval_every: int = 500
	sample_every: int = 500
	num_samples: int = 64


@dataclass
class MetricsConfig:
	batch_size_to_generate: int
	num_samples: int
	regression_steps: Sequence[int]
