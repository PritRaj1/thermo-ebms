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


class ImportanceTuner:
	def __init__(self, resampler_type: str):
		self.resampler = resampler_map.get(resampler_type, "residual")

	def logllhood(self, model: nnx.Module, z: jax.Array, x: jax.Array) -> jax.Array:
		x_pred = model.gen(z)
		return jnp.sum(
			(x[:, None, ...] - x_pred[None, :, ...]) ** 2,
			axis=tuple(range(2, x.ndim + 1)),
		)

	def batch_resample(self, key: jax.Array, logllhood: jax.Array, N: int):
		subkeys = jax.random.split(key, N)
		weights = jax.nn.softmax(logllhood, axis=-1)
		return jax.vmap(self.resampler, in_axes=(0, 0, None))(subkeys, weights, N)

	def __call__(self, key: jax.Array, model: nnx.Module, x: jax.Array) -> jax.Array:
		num_samples = x.shape[0]
		key, prior_key, posterior_key = jax.random.split(key, 3)

		z_prior = model.sample_prior(prior_key, num_samples)
		ll = self.logllhood(model, z_prior, x)
		idx = self.batch_resample(posterior_key, ll, num_samples)
		z_post = z_prior[idx].reshape(-1, *z_prior.shape[1:])
		x = jnp.repeat(x, num_samples, axis=0)

		contrastive_div = model.ebm.loss(z_post, z_prior) / (num_samples**2)
		recon = model.gen.loss(x, z_post) / (num_samples**2)
		return contrastive_div + recon
