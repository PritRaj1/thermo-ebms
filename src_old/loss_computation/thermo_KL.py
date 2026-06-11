import jax.numpy as jnp
from jax.scipy.linalg import det, inv
import configparser
from jax.nn import softmax

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])


def KL_div(z1, z2, eps=1e-8, ridge=1e-5):
    """Analytic solution for KL Divergence between power posteriors, assuming multivariate Gaussians."""
    m1 = jnp.mean(z1, axis=0)
    m2 = jnp.mean(z2, axis=0)
    var1 = jnp.cov(z1, rowvar=False) + ridge * jnp.eye(z_channels)
    var2 = jnp.cov(z2, rowvar=False) + ridge * jnp.eye(z_channels)

    KL_div = 0.5 * (
        jnp.log((det(var2) + eps) / (det(var1) + eps))
        + jnp.trace(inv(var2) @ var1)
        + (m2 - m1).T @ inv(var2) @ (m2 - m1)
        - z_channels
    )

    return KL_div


def analytic_KL_bias(z_prev, z_curr):
    """Returns the KL divergence bias term between two adjacent temperatures."""

    z_prev = z_prev.squeeze()
    z_curr = z_curr.squeeze()
    z_prev = softmax(z_prev, axis=-1)
    z_curr = softmax(z_curr, axis=-1)
    return KL_div(z_prev, z_curr) - KL_div(z_curr, z_prev)
