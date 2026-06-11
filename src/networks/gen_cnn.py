import jax
from flax import nnx
from ml_collections import ConfigDict


class GEN(nnx.Module):
	def __init__(self, gen_config: ConfigDict, rngs: nnx.Rngs):
		input_dim = gen_config.z_dim
		hidden_dim = gen_config.hidden_dim
		output_dim = gen_config.img_channels
		image_dim = gen_config.image_res
		leak_coef = gen_config.leakyrelu_leak
		self.sigma = gen_config.gaussian_stddev

		act = lambda x: nnx.leaky_relu(
			x,
			negative_slope=leak_coef,
		)

		deconv = lambda cin, cout, stride=2, padding="SAME": nnx.ConvTranspose(
			in_features=cin,
			out_features=cout,
			kernel_size=(4, 4),
			strides=(stride, stride),
			padding=padding,
			rngs=rngs,
		)

		bn = lambda c: nnx.BatchNorm(
			num_features=c,
			momentum=0.9,
			epsilon=1e-5,
			rngs=rngs,
		)

		block = lambda cin, cout: [
			deconv(cin, cout),
			bn(cout),
			act,
		]

		layers = [
			deconv(
				input_dim,
				hidden_dim * 16,
				stride=1,
				padding="VALID",
			),
			bn(hidden_dim * 16),
			act,
			*block(hidden_dim * 16, hidden_dim * 8),
			*block(hidden_dim * 8, hidden_dim * 4),
		]

		# CELEBA in 64x64; CIFAR10, SVHN in 32x32
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

	def loolkhood(self, z, x, t):
		"""log[ p_β(x | z)^t ] ∝ t * [ - (x - g(z))^2 / (2 * σ^2) ]"""
		sqr_err = (x - self(z)) ** 2
		return -t * (sqr_err) / (2 * self.sigma**2)
