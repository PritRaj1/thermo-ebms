from dataclasses import dataclass, field

from .model import KAEMConfig, ModelConfig, ThermoConfig
from .networks import ConvBlock, EBMConfig, GENConfig, HMCConfig, ULAConfig
from .pipeline import (
	AdamWConfig,
	LoggingConfig,
	MetricsConfig,
	OptConfig,
	TrainingConfig,
)
from .types import ScoreFn, XchangeFn


@dataclass
class Config:
	model: ModelConfig = field(default_factory=ModelConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	logging: LoggingConfig = field(default_factory=LoggingConfig)
	unbiased_metrics: MetricsConfig = field(default_factory=MetricsConfig)
	optim: OptConfig = field(default_factory=OptConfig)


__all__ = [
	"AdamWConfig",
	"Config",
	"ConvBlock",
	"EBMConfig",
	"GENConfig",
	"HMCConfig",
	"KAEMConfig",
	"ModelConfig",
	"OptConfig",
	"ScoreFn",
	"ThermoConfig",
	"TrainingConfig",
	"ULAConfig",
	"XchangeFn",
]
