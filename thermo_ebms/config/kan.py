from dataclasses import dataclass, field
from typing import Literal
from flax import nnx


BasisType = Literal["rbf", "spline", "chebyshev", "fourier"]


@dataclass
class RBFConfig:
	D: int = 8
	kernel: dict | None = None
	grid_e: float = 1.0
	grid_range: tuple[float, float] = (-1.0, 1.0)
	residual: nnx.Module | None = nnx.relu
	add_bias: bool = True


@dataclass
class SplineConfig:
	k: int = 3
	G: int = 5
	grid_e: float = 1.0
	grid_range: tuple[float, float] = (-1.0, 1.0)
	residual: nnx.Module | None = nnx.relu
	add_bias: bool = True


@dataclass
class ChebyshevConfig:
	D: int = 3
	flavor: str | None = "modified"
	residual: nnx.Module | None = nnx.relu
	add_bias: bool = True


@dataclass
class FourierConfig:
	D: int = 3
	smooth_init: bool = True
	add_bias: bool = True


@dataclass
class GridUpdatingConfig:
	numgrid: int = 8
	update_frequency: int = 100
	frequency_decay: float = 0.999


@dataclass
class KANConfig:
	basis: BasisType = "rbf"
	rbf: RBFConfig | None = None
	spline: SplineConfig | None = None
	chebyshev: ChebyshevConfig | None = None
	fourier: FourierConfig | None = None
	grid_updating: GridUpdatingConfig = field(default_factory=GridUpdatingConfig)
