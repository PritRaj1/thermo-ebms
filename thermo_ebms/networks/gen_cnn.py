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
		hidden_dim = gen_config.hidden_dim
		output_dim = gen_config.img_channels
		image_dim = gen_config.image_res
		self.sigma = gen_config.gaussian_stddev

		act = lambda x: nnx.leaky_relu(x, negative_slope=gen_config.leakyrelu_leak)

		def deconv(cin, cout, stride=2, padding="SAME"):
			return nnx.ConvTranspose(
				in_features=cin,
				out_features=cout,
				kernel_size=(4, 4),
				strides=(stride, stride),
				padding=padding,
				rngs=rngs,
			)

		def bn(c):
			return nnx.BatchNorm(
				num_features=c,
				momentum=0.9,
				epsilon=1e-5,
				rngs=rngs,
			)

		def block(cin, cout):
			return [
				deconv(cin, cout),
				bn(cout),
				act,
			]

		layers = [
			deconv(z_dim, hidden_dim * 16, stride=1, padding="VALID"),
			bn(hidden_dim * 16),
			act,
			*block(hidden_dim * 16, hidden_dim * 8),
			*block(hidden_dim * 8, hidden_dim * 4),
		]

		if image_dim == 64:
			layers.extend(
				[
					*block(hidden_dim * 4, hidden_dim * 2),
					deconv(hidden_dim * 2, output_dim),
					bn(output_dim),
				]
			)
		else:
			layers.extend(
				[
					deconv(hidden_dim * 4, output_dim),
					bn(output_dim),
				]
			)

		layers.append(jax.nn.tanh)

		self.g = nnx.Sequential(*layers)

	def __call__(self, z: jax.Array) -> jax.Array:
		return self.g(z)

	def loglkhood(
		self,
		z: jax.Array,
		x: jax.Array,
	) -> jax.Array:
		"""log p(x|z)^t ∝ -t * ||x - g(z)||^2 / (2σ^2)"""
		diff = x - self(z)
		sqr_err = diff**2
		ll = sqr_err / (2 * self.sigma**2)
		return -ll.sum()

	def loss(self, x: jax.Array, z_post: jax.Array) -> jax.Array:
		"""Gaussian/pixel loss"""
		return ((x - self(z_post)) ** 2).mean()
