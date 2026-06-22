import jax
from flax import nnx
import optax


@nnx.jit(static_argnums=(0,))
def update(
	tx: nnx.Optimizer,
	opt_st: nnx.OptState,
	model: nnx.Module,
	x: jax.Array,
	z_post: jax.Array,
	z_prior: jax.Array,
) -> jax.Array:

	def loss_fn(m):
		cd = m.ebm.loss(z_post, z_prior)
		recon = m.loss(x, z_post, z_prior)
		return cd + recon

	loss_val, grads = nnx.value_and_grad(loss_fn)(model)
	graph, ps, st = nnx.split(model, nnx.Param, ...)
	updates, new_opt_st = tx.update(grads, opt_st, ps)
	new_ps = optax.apply_updates(ps, updates)
	new_model = nnx.merge(graph, new_ps, st)
	new_model.train_idx += 1
	return new_model, new_opt_st, loss_val


def train_step(
	tx: nnx.Optimizer,
	opt_st: nnx.OptState,
	model: nnx.Module,
	x: jax.Array,
	key: jax.Array,
) -> tuple[nnx.Module, nnx.OptState, jax.Array, jax.Array]:
	key, prior_key, posterior_key = jax.random.split(key, 3)
	z_prior = model.sample_prior(prior_key, x.shape[0])
	z_post = model.sample_posterior(posterior_key, x)

	if model.num_temps > 1:
		model.adapt_temps(x, z_post)

	if (model.base == "kaem") and hasattr(model.ebm.f.layers[0], "grid"):
		model.update_grid(z_post)

	model.train()
	new_model, new_st, loss = update(tx, opt_st, model, x, z_post, z_prior)
	return new_model, new_st, loss, key


@nnx.jit
def mse(x: jax.Array, x_gen: jax.Array) -> jax.Array:
	return ((x - x_gen) ** 2).mean()


def eval_step(
	model: nnx.Module, x: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
	x_gen, key = model(key, x.shape[0])
	return mse(x, x_gen), key
