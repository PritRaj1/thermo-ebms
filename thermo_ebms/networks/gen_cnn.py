import jax
from flax import nnx
from ml_collections import ConfigDict


class GEN(nnx.Module):
	def __init__(
		self,
		gen_config: ConfigDict,
		z_dim: int,
		rngs: nnx.Rngs,
	):
		channels = gen_config.cnn_channels
		kernel_sizes = gen_config.kernel_sizes
		strides = gen_config.strides
		paddings = gen_config.paddings
		output_dim = gen_config.img_channels
		self.sigma = gen_config.gaussian_stddev

		assert len(kernel_sizes) == len(strides) == len(paddings) == len(channels), (
			f"Config mismatch: "
			f"len(kernel_sizes)={len(kernel_sizes)}, "
			f"len(strides)={len(strides)}, "
			f"len(paddings)={len(paddings)}, "
			f"len(cnn_channels)={len(channels)}"
		)

		def act(x: jax.Array) -> jax.Array:
			return nnx.leaky_relu(x, negative_slope=gen_config.leakyrelu_leak)

		def deconv(cin, cout, k, s, p):
			return nnx.ConvTranspose(
				in_features=cin,
				out_features=cout,
				kernel_size=k,
				strides=s,
				padding=p,
				rngs=rngs,
			)

		def bn(c):
			return nnx.BatchNorm(
				num_features=c,
				momentum=0.9,
				epsilon=1e-5,
				rngs=rngs,
			)

		layers = []
		layers.extend(
			[
				deconv(z_dim, channels[0], kernel_sizes[0], strides[0], paddings[0]),
				bn(channels[0]),
				act,
			]
		)

		for i in range(len(channels) - 1):
			layers.extend(
				[
					deconv(
						channels[i],
						channels[i + 1],
						kernel_sizes[i + 1],
						strides[i + 1],
						paddings[i + 1],
					),
					bn(channels[i + 1]),
					act,
				]
			)

		layers.extend(
			[
				deconv(
					channels[-1],
					output_dim,
					kernel_sizes[-1],
					strides[-1],
					paddings[-1],
				),
				bn(output_dim),
				jax.nn.tanh,
			]
		)

		self.g = nnx.Sequential(*layers)

	def __call__(self, z: jax.Array) -> jax.Array:
		return self.g(z)

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
		return - grad_ll / (2 * self.sigma**2)
