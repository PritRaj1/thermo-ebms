from dataclasses import dataclass, field

from .networks import MCMCConfig, ConvBlock, EBMConfig, GENConfig
from .types import ScoreFn, XchangeFn
from .model import ThermoConfig, KAEMConfig, ModelConfig
from .pipeline import (
	TrainingConfig,
	LoggingConfig,
	MetricsConfig,
	AdamConfig,
	OptConfig,
)


@dataclass
class Config:
	model: ModelConfig = field(default_factory=ModelConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	logging: LoggingConfig = field(default_factory=LoggingConfig)
	unbiased_metrics: MetricsConfig = field(default_factory=MetricsConfig)
	optim: OptConfig = field(default_factory=OptConfig)


__all__ = [
	"EBMConfig",
	"GENConfig",
	"MCMCConfig",
	"AdamConfig",
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
