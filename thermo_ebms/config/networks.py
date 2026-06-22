from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Literal


@dataclass
class MCMCConfig:
	stepsize: float = 0.01
	numsteps: int = 60


@dataclass
class OptConfig:
	lr_init: float = 0.0001
	lr_end: float = 0.00002
	lr_decay: float = 0.998
	beta1: float = 0.999
	beta2: float = 0.5
	begin: int = 0
	step: int = 0


@dataclass
class EBMConfig:
	p0_stddev: float = 1.0
	leakyrelu_leak: float = 0.1
	mcmc: MCMCConfig = field(default_factory=MCMCConfig)
	layer_widths: list[int] = field(default_factory=lambda: [200, 200, 1])


@dataclass
class ConvBlock:
	channels: int = 32
	kernel_size: int = 4
	stride: int = 1
	padding: Literal["SAME", "VALID"] = "SAME"


@dataclass
class GENConfig:
	img_channels: int = 3
	image_res: int = 32
	gaussian_stddev: float = 0.3
	leakyrelu_leak: float = 0.2
	mcmc: MCMCConfig = field(default_factory=MCMCConfig)
	blocks: Sequence[ConvBlock] = field(
		default_factory=lambda: [
			ConvBlock(
				channels=16,
				kernel_size=4,
				stride=1,
				padding="VALID",
			),
			ConvBlock(
				channels=8,
				kernel_size=4,
				stride=2,
				padding="SAME",
			),
			ConvBlock(
				channels=4,
				kernel_size=4,
				stride=2,
				padding="SAME",
			),
		]
	)
