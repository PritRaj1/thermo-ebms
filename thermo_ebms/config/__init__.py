from dataclasses import dataclass, field

from .model import KAEMConfig, ModelConfig, ThermoConfig
from .networks import ConvBlock, EBMConfig, GENConfig, MCMCConfig
from .pipeline import (
	AdamConfig,
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
	"AdamConfig",
	"Config",
	"ConvBlock",
	"EBMConfig",
	"GENConfig",
	"KAEMConfig",
	"MCMCConfig",
	"ModelConfig",
	"OptConfig",
	"ScoreFn",
	"ThermoConfig",
	"TrainingConfig",
	"XchangeFn",
]
