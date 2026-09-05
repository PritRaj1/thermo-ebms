import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from flax import nnx

resampler_map = {
	"residual": resampling.residual,
	"systematic": resampling.systematic,
	"stratified": resampling.stratified,
	"multinomial": resampling.multinomial,
}


def pairwise_sqdist(xi: jax.Array, xj: jax.Array) -> jax.Array:
	return jnp.sum((xi - xj) ** 2)


class ImportanceTuner(nnx.Module):
	def __init__(self, resampler_type: str):
		self.resampler = resampler_map.get(resampler_type, "residual")

	def logllhood(self, model: nnx.Module, z: jax.Array, x: jax.Array) -> jax.Array:
		x_pred = model.gen(z)
		return jnp.sum(
			(x[:, None, ...] - x_pred[None, :, ...]) ** 2,
			axis=tuple(range(2, x.ndim + 1)),
		)

	@nnx.jit
	def batch_resample(
		self, subkeys: jax.Array, model: nnx.Module, z: jax.Array, x: jax.Array
	) -> tuple[jax.Array, jax.Array, jax.Array]:
		def resample_one(key, weights):
			return self.resampler(key, weights, N)

		N = z.shape[0]
		ll = self.logllhood(model, z, x)
		weights = jax.nn.softmax(ll, axis=-1)
		return jax.vmap(resample_one)(subkeys, weights)

	def __call__(
		self, model: jax.Array, x: jax.Array, z_post: jax.Array, z_prior: jax.Array
	) -> jax.Array:
		num_samples = x.shape[0]
		contrastive_div = model.ebm.loss(z_post, z_prior) / num_samples
		recon = model.gen.loss(x, z_post) / num_samples
		return contrastive_div + recon
