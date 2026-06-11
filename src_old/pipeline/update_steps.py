import jax
import jax.numpy as jnp
import optax

def update_params(optimiser_tup, grad_list, opt_state_tup, params_tup):
    """
    Function to update the parameters of the models.

    Args:
    - optimiser_tup: tuple of optimisers
    - grad_list: list of gradients
    - opt_state_tup: tuple of optimiser states
    - params_tup: tuple of model parameters

    Returns:
    - new_params_set: tuple of updated model parameters
    - new_opt_states: tuple of updated optimiser states
    """

    ebm_updates, new_ebm_opt_state = optimiser_tup[0].update(
        grad_list[0], opt_state_tup[0]
    )

    gen_updates, new_gen_opt_state = optimiser_tup[1].update(
        grad_list[1], opt_state_tup[1]
    )

    new_ebm_params = optax.apply_updates(params_tup[0], ebm_updates)
    new_gen_params = optax.apply_updates(params_tup[1], gen_updates)

    return (new_ebm_params, new_gen_params), (new_ebm_opt_state, new_gen_opt_state)


def get_grad_stats(grad_ebm, grad_gen):

    # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.mean(jnp.concatenate([grad_ebm, grad_gen])), jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
