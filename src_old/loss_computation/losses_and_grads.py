import configparser
from jax import value_and_grad
from jax.lax import stop_gradient

from src.pipeline.update_steps import *
from src.loss_computation.vanilla_computation import vanilla_loss
from src.loss_computation.thermo_computation import thermo_loss

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")
temp_power = float(parser["TEMP"]["TEMP_POWER"])

if temp_power > 0:
    print("Using Temperature Schedule with Power: {}".format(temp_power))
    loss_fcn = thermo_loss
else:
    loss_fcn = vanilla_loss
    print("Using no Thermodynamic Integration, defaulting to Vanilla Model")


def get_losses_and_grads(key, x, params_tup, fwd_fcn_tup):
    """Function to compute the losses and gradients during training."""

    ebm_params, gen_params = params_tup

    # Compute loss of both models
    (total_loss, key), (grad_ebm, grad_gen) = value_and_grad(
        loss_fcn, argnums=(2, 3), has_aux=True
    )(key, stop_gradient(x), ebm_params, gen_params, *fwd_fcn_tup)

    return key, total_loss, grad_ebm, grad_gen


def get_loss(key, x, params_tup, fwd_fcn_tup):
    """Function to compute the loss during validation."""

    ebm_params, gen_params = params_tup

    # Compute loss of both models
    total_loss, key = loss_fcn(
        key,
        stop_gradient(x),
        stop_gradient(ebm_params),
        stop_gradient(gen_params),
        *fwd_fcn_tup
    )

    return key, total_loss
