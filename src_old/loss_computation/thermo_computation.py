import jax
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
import configparser

from src.loss_computation.loss_helper_fcns import batch_sample_posterior, llhood
from src.loss_computation.thermo_KL import analytic_KL_bias

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])
beta = float(parser["TEMP"]["KL_BIAS_WEIGHT"])

temp_schedule = jnp.linspace(0, 1, num_temps) ** temp_power
batched_llhood = jax.vmap(llhood, in_axes=(0, 0, None, None, None))

# Determine which bias term to use
if beta != 0:
    get_bias = lambda *args: beta * analytic_KL_bias(*args)
else:
    get_bias = lambda *args: 0.0


def thermo_scan_loop(carry, t, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Scan step to compute the expected llhood loss at one temperature."""

    # Parse the carry state
    key, t_prev, prev_loss, prev_z, keep_KL = carry

    # Get liklihood, E_{z|x,t}[ log(p_β(x|z)) ]
    key, z_posterior = batch_sample_posterior(
        key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )
    current_loss = batched_llhood(z_posterior, x, 1.0, GEN_params, GEN_fwd).mean()

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # ((L(t_i) + L(t_{i-1})) * ∇T) + (KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}])
    temperature_loss = (current_loss + prev_loss) * delta_T + get_bias(
        prev_z, z_posterior
    ) * keep_KL  # Do not include KL divergence in first iter, (area is 0 between t=0 and t=0)

    # Push tempered loss to the stack and carry over the current state
    return (key, t, current_loss, z_posterior, 1), temperature_loss


def thermo_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Function to compute the themodynamic integration loss for the GEN model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:

    log p_θ(x) =
            1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
            + 1/2 * Σ [ KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}] ] ]

    The terms within the summations are accumulated in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.
    The KL Divergence term is included to account for bias.

    Returns:
    The total lkhood loss for the entire thermodynamic integration loop:
    -∫ E_{z|x,t}[ log(p_β(x|z)) ] dt
    """

    # Wrap the themodynamic loop in a partial function to exploit partial immutability
    scan_loss = partial(
        thermo_scan_loop,
        x=x,
        EBM_params=EBM_params,
        GEN_params=GEN_params,
        EBM_fwd=EBM_fwd,
        GEN_fwd=GEN_fwd,
    )

    # Scan along each temperature and stack the losses
    z_init = batch_sample_posterior(
        key, x, 0.0, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )[1]
    initial_state = (key, 0.0, 0.0, z_init, 0)
    (key, _, _, _, _), temp_losses = scan(
        f=scan_loss, init=initial_state, xs=temp_schedule
    )

    return -0.5 * temp_losses.sum(), key
