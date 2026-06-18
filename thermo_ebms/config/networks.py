from dataclasses import dataclass
from collections.abc import Sequence


@dataclass
class MCMCConfig:
	stepsize: float = 0.01
	numsteps: int = 60


class OptConfig:
	lr_init: float = 0.001
	lr_end: float = 0.0001
	lr_decay: float = 0.975
	beta1: float = 0.999
	beta2: float = 0.5
	begin: int = 0
	step: int = 0


@dataclass
class EBMConfig:
	layer_widths: list[int]
	p0_stddev: float = 1.0
	leakyrelu_leak: float = 0.1
	mcmc: MCMCConfig | None = None


@dataclass
class ConvBlock:
	channels: int
	kernel_size: int
	stride: int
	padding: int


@dataclass
class GENConfig:
	blocks: Sequence[ConvBlock]
	img_channels: int = 3
	image_res: int = 32
	gaussian_stddev: float = 0.3
	leakyrelu_leak: float = 0.2
	mcmc: MCMCConfig | None = None
