from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Literal


@dataclass
class MCMCConfig:
	stepsize: float = 0.01
	numsteps: int = 60


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
	gaussian_stddev: float = 0.3
	leakyrelu_leak: float = 0.2
	groupnorm: bool = False
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
