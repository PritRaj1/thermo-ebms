from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ULAConfig:
	stepsize: float = 0.01
	numsteps: int = 60


class HMCConfig:
	stepsize: float = 0.01
	num_integration: int = 10


@dataclass
class EBMConfig:
	p0_stddev: float = 1.0
	leakyrelu_leak: float = 0.1
	mcmc: ULAConfig = field(default_factory=ULAConfig)
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
	mixed_precision: bool = True
	gaussian_stddev: float = 0.3
	groupnorm: bool = False
	mcmc: HMCConfig = field(default_factory=HMCConfig)
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
