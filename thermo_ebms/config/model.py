from dataclasses import dataclass, field

from .networks import EBMConfig, GENConfig


@dataclass
class KAEMConfig:
	mixture: bool = True
	numquad: int = 25
	p0_stddev: float = 1.0
	numcentres: int = 10
	mixture_regularization: float = 0.0001


@dataclass
class ThermoConfig:
	num_temps: int = 1
	xchange_every: int = 0
	annealing_cycle: int = 0
	powerlaw_start: int = 6
	powerlaw_end: int = 6


@dataclass
class ModelConfig:
	seed: int = 0
	z_dim: int = 100
	ebm: EBMConfig = field(default_factory=EBMConfig)
	gen: GENConfig = field(default_factory=GENConfig)
	kaem: KAEMConfig = field(default_factory=KAEMConfig)
	thermo: ThermoConfig = field(default_factory=ThermoConfig)
