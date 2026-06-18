import jax
import jax.numpy as jnp
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


@nnx.jit
def _eval(
	model: nnx.Module, x: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
	key, subkey = jax.random.split(key)
	z_prior = model.sample_prior(subkey, x.shape[0])
	return model.gen.loss(x, z_prior), key


def train_step(
	tx: nnx.Optimizer,
	opt_st: nnx.OptState,
	model: nnx.Module,
	x: jax.Array,
	key: jax.Array,
) -> tuple[nnx.Module, nnx.OptState, jax.Array, jax.Array]:
	model.eval()

	# Sample prior, kaem requires mixture selection
	if model.base == "kaem":
		key = model.sample_mixture(key, x.shape[0])

	key, prior_key, posterior_key = jax.random.split(key, 3)
	z_prior = model.sample_prior(prior_key, x.shape[0])

	# Sample posterior, handle expanded z dim
	mixture_component = model.component
	if (model.base == "kaem") and (model.num_temps > 1):
		model.component = jnp.repeat(model.component, model.num_temps, axis=0)

	z_post = model.sample_posterior(posterior_key, x)

	# Train step with annealing schedule adaption
	model.component = mixture_component
	model.train()
	new_model, new_st, loss = update(tx, opt_st, model, x, z_post, z_prior)

	new_model.eval()
	if new_model.num_temps > 1:
		new_model.adapt_temps(x, z_post)

	return new_model, new_st, loss, key


def gen(model: nnx.Module, N: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
	model.eval()
	if model.base == "kaem":
		key = model.sample_mixture(key, N)

	return model(key, N)


def eval_step(
	model: nnx.Module, x: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
	model.eval()
	if model.base == "kaem":
		key = model.sample_mixture(key, x.shape[0])

	return _eval(model, x, key)
