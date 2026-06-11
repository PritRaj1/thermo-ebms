import jax

from src.pipeline.update_steps import *
from src.loss_computation.losses_and_grads import get_losses_and_grads, get_loss

def train_step(key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup):
    """Single parameter update for the EBM and GEN models."""

    key, subkey = jax.random.split(key)
    key, total_loss, grad_e, grad_g = get_losses_and_grads(
        subkey, x, params_tup, fwd_fcn_tup
    )

    params_tup, opt_state_tup = update_params(
        optimiser_tup, [grad_e, grad_g], opt_state_tup, params_tup
    )

    grad_mean, grad_var = get_grad_stats(grad_e, grad_g)

    return key, params_tup, opt_state_tup, total_loss, grad_mean, grad_var


def val_step(key, x, params_tup, fwd_fcn_tup):
    """Single evaluation on an unseen image set."""

    key, subkey = jax.random.split(key)
    key, total_loss = get_loss(subkey, x, params_tup, fwd_fcn_tup)

    return key, total_loss
