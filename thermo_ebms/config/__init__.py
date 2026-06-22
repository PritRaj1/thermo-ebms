from dataclasses import dataclass, field

from .kan import KANConfig
from .networks import MCMCConfig, OptConfig, ConvBlock, EBMConfig, GENConfig
from .types import ScoreFn, XchangeFn
from .model import ThermoConfig, KAEMConfig, ModelConfig
from .pipeline import TrainingConfig, LoggingConfig, MetricsConfig, ScheduleConfig


@dataclass
class Config:
	model: ModelConfig = field(default_factory=ModelConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	logging: LoggingConfig = field(default_factory=LoggingConfig)
	unbiased_metrics: MetricsConfig = field(default_factory=MetricsConfig)
	lr_schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


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
	"ScheduleConfig",
	"Config",
]
