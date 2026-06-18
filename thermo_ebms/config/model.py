from dataclasses import dataclass
from typing import Literal

from .networks import EBMConfig, GENConfig
from .kan import KANConfig


@dataclass
class KAEMConfig:
	kan: KANConfig
	mixture: bool = True
	numquad: int = 25
	numgrid: int = 50


@dataclass
class ThermoConfig:
	num_temps: int = 1
	xchange_every: int = 0


@dataclass
class ScheduleConfig:
	begin: int = 0
	step: int = 0


@dataclass
class ModelConfig:
	seed: int
	z_dim: int
	base: Literal["neural", "kaem"]
	ebm: EBMConfig
	gen: GENConfig
	kaem: KAEMConfig
	thermo: ThermoConfig
	opt_schedule: ScheduleConfig
