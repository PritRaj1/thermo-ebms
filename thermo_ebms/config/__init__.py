from dataclasses import dataclass

from .kan import KANConfig
from .networks import MCMCConfig, OptConfig, ConvBlock, EBMConfig, GENConfig
from .types import ScoreFn, XchangeFn
from .model import ThermoConfig, KAEMConfig, ModelConfig
from .pipeline import (
	TrainingConfig,
	LoggingConfig,
	MetricsConfig,
)


@dataclass
class Config:
	model: ModelConfig
	training: TrainingConfig
	logging: LoggingConfig
	unbiased_metrics: MetricsConfig


__all__ = [
	"KANConfig",
	"EBMConfig",
	"GENConfig",
	"MCMCConfig",
	"OptConfig",
	"ConvBlock",
	"ThermoConfig",
	"ScoreFn",
	"XchangeFn",
	"KAEMConfig",
	"ModelConfig",
	"TrainingConfig",
	"Config",
]
