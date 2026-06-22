from dataclasses import dataclass, field
from typing import Literal

from .networks import EBMConfig, GENConfig
from .kan import KANConfig


@dataclass
class KAEMConfig:
	kan: KANConfig = field(default_factory=KANConfig)
	mixture: bool = True
	numquad: int = 25
	numgrid: int = 10


@dataclass
class ThermoConfig:
	num_temps: int = 1
	xchange_every: int = 0


@dataclass
class ModelConfig:
	seed: int = 0
	z_dim: int = 100
	base: Literal["neural", "kaem"] = "kaem"
	ebm: EBMConfig = field(default_factory=EBMConfig)
	gen: GENConfig = field(default_factory=GENConfig)
	kaem: KAEMConfig = field(default_factory=KAEMConfig)
	thermo: ThermoConfig = field(default_factory=ThermoConfig)
