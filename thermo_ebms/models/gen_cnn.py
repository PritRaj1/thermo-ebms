import jax
import jax.numpy as jnp
from flax import nnx

from ..config import GENConfig, ConvBlock


class GEN(nnx.Module):
	half_prec = jnp.bfloat16
	full_prec = jnp.float32

	def __init__(self, config: GENConfig, z_dim: int, rngs: nnx.Rngs):
		self.sigma = config.gaussian_stddev

		def act(x):
			return nnx.leaky_relu(x, negative_slope=config.leakyrelu_leak)

		def deconv(cin, block: ConvBlock):
			return nnx.ConvTranspose(
				in_features=cin,
				out_features=block.channels,
				kernel_size=(block.kernel_size, block.kernel_size),
				strides=(block.stride, block.stride),
				padding=block.padding,
				rngs=rngs,
				param_dtype=self.full_prec,
				dtype=self.half_prec,
			)

		def bn(c):
			return nnx.BatchNorm(
				num_features=c,
				momentum=0.9,
				epsilon=1e-5,
				rngs=rngs,
				param_dtype=self.full_prec,
				dtype=self.half_prec,
			)

		layers = []
		first = config.blocks[0]
		layers += [
			deconv(z_dim, first),
			bn(first.channels),
			act,
		]

		for prev, block in zip(config.blocks[:-1], config.blocks[1:]):
			layers += [
				deconv(prev.channels, block),
				bn(block.channels),
				act,
			]

		last = config.blocks[-1]
		layers += [
			deconv(
				last.channels,
				ConvBlock(
					channels=config.img_channels,
					kernel_size=last.kernel_size,
					stride=last.stride,
					padding=last.padding,
				),
			),
			jax.nn.tanh,
		]

		self.g = nnx.Sequential(*layers)

	def __call__(self, z: jax.Array) -> jax.Array:
		z = z.sum(axis=2, keepdims=True)
		z = z.astype(self.half_prec)
		return self.g(z).astype(self.full_prec)

	def loss(self, x: jax.Array, z_post: jax.Array) -> jax.Array:
		"""Gaussian/pixel loss"""
		return ((x - self(z_post)) ** 2).mean()

	def posterior_score(
		self,
		z: jax.Array,
		x: jax.Array,
	) -> jax.Array:
		"""∇_z log p(x|z)^t ∝ ||x - g(z)||^2 / (2σ^2)"""

		def wrapped_ll(z_i: jax.Array) -> jax.Array:
			return self.loss(x, z_i)

		grad_ll = jax.grad(wrapped_ll)(z)
		return -grad_ll / (2 * self.sigma**2)
