import jax
from functools import partial
import configparser
from jax.lax import scan, stop_gradient

from src.MCMC_Samplers.langevin_updates import langevin_prior, langevin_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
prior_steps = int(parser["MCMC"]["E_SAMPLE_STEPS"])
prior_s = float(parser["MCMC"]["E_STEP_SIZE"])
posterior_steps = int(parser["MCMC"]["G_SAMPLE_STEPS"])
posterior_s = float(parser["MCMC"]["G_STEP_SIZE"])
kill_gradient = bool(parser["MCMC"]["KILL_GRADIENT"])

if kill_gradient:
    final_sample_fcn = lambda x: stop_gradient(x)
else:
    final_sample_fcn = lambda x: x

def sample_p0(key):
    """Sample from the noise prior distribution."""

    key, subkey = jax.random.split(key)
    return key, p0_sig * jax.random.normal(subkey, (1, 1, z_channels))


def get_noise_step(key, num_steps, shape):
    """Get all noises for the MCMC steps."""

    key, subkey = jax.random.split(key)
    return key, jax.random.normal(subkey, (num_steps,) + shape)


def sample_prior(key, EBM_params, EBM_fwd):
    """
    Sample from the exponentially-tilted prior distribution.

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_α(x)
    """

    # Wrap the langevin update function with constants for scanning
    scan_MCMC = partial(langevin_prior, EBM_params=EBM_params, EBM_fwd=EBM_fwd)

    # Sample all noise at once, to avoid reseeding the PRNG and reduce overhead
    key, z0 = sample_p0(key)
    key, noise = get_noise_step(key, prior_steps, z0.shape)

    # Scan along the noise to iteratively update z
    z_prior, _ = scan(scan_MCMC, z0, noise, length=prior_steps)

    return key, final_sample_fcn(z_prior)


def sample_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Sample from the posterior distribution.

    Returns:
    - key: PRNG key
    - z_samples: samples from the tempered posterior distribution, p_θ(z|x,t)
    """

    # Wrap the langevin update function with constants for scanning
    scan_MCMC = partial(
        langevin_posterior,
        x=x,
        t=t,
        EBM_params=EBM_params,
        GEN_params=GEN_params,
        EBM_fwd=EBM_fwd,
        GEN_fwd=GEN_fwd,
    )

    # Sample all noise at once, to avoid reseeding the PRNG and reduce overhead
    key, z0 = sample_p0(key)
    key, noise = get_noise_step(key, posterior_steps, z0.shape)

    # Scan along the noise to iteratively update z
    z_posterior, _ = scan(scan_MCMC, z0, noise, length=posterior_steps)

    return final_sample_fcn(z_posterior)
