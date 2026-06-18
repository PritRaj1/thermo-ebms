from dataclasses import dataclass
from typing import Literal


BasisType = Literal["rbf", "spline", "chebyshev", "fourier"]


@dataclass
class RBFConfig:
	sigma: float = 1.0


@dataclass
class SplineConfig:
	k: int
	G: int


@dataclass
class ChebyshevConfig:
	degree: int


@dataclass
class FourierConfig:
	modes: int


@dataclass
class KANConfig:
	basis: BasisType
	rbf: RBFConfig | None = None
	spline: SplineConfig | None = None
	chebyshev: ChebyshevConfig | None = None
	fourier: FourierConfig | None = None
