from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class AdamWConfig:
	lr_init: float = 0.0001
	lr_end: float = 0.00002
	weight_decay: float = 0.0001
	beta1: float = 0.999
	beta2: float = 0.9
	warmup_fraction: float = 0.2


@dataclass
class OptConfig:
	ebm: AdamWConfig = field(default_factory=AdamWConfig)
	gen: AdamWConfig = field(default_factory=AdamWConfig)
	kan: AdamWConfig = field(default_factory=AdamWConfig)


@dataclass
class ImportanceConfig:
	type: str = "residual"
	epochs: int = 100
	optim: OptConfig = field(default_factory=OptConfig)


@dataclass
class TrainingConfig:
	dataset: str = "cifar10"
	epochs: int = 100
	global_batch_size: int = 128
	image_res: int = 32
	is_finetune: ImportanceConfig = field(default_factory=ImportanceConfig)


@dataclass
class LoggingConfig:
	logdir: str = "./logs"
	ckpt_dir: str = "./checkpoints"
	ckpt_every: int = 20
	sample_every: int = 10
	num_samples: int = 128


@dataclass
class MetricsConfig:
	batch_size_to_generate: int = 200
	num_samples: int = 20000
	regression_steps: Sequence[int] = (
		2000,
		4000,
		6000,
		8000,
		10000,
		12000,
		14000,
		16000,
		18000,
		20000,
	)
