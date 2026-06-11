from jax.scipy.special import logsumexp
from jax.lax import stop_gradient
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


def log_prior_fcn(z, EBM_params, EBM_fwd):
    """Compute log(p_α(z)) ∝ f(z) - 0.5 * (z^2) / (σ^2)"""
    return EBM_fwd(EBM_params, z) - 0.5 * (z**2) / (p0_sig**2)


def log_llood_fcn(z, x, t, GEN_params, GEN_fwd):
    """Compute log[ p_β(x | z)^t ] ∝ t * [ - (x - g(z))^2 / (2 * σ^2) ]"""
    g_z = GEN_fwd(GEN_params, z)
    sqr_err = (x - g_z) ** 2
    return -t * (sqr_err) / (2 * pl_sig**2)
