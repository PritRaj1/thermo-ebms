import jax
from typing import Protocol


class ScoreFn(Protocol):
	"""Returns score of a PDF, ∇_z log(PDF(z))"""

	def __call__(self, z: jax.Array) -> jax.Array: ...


class XchangeFn(Protocol):
	"""Returns swapped samples between power posteriors"""

	def __call__(
		self,
		key: jax.Array,
		z: jax.Array,
		idx: jax.Array,
	) -> jax.Array: ...
