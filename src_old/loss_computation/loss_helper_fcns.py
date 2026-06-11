import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import configparser

from src.MCMC_Samplers.sample_distributions import sample_posterior, sample_prior
from src.MCMC_Samplers.log_pdfs import log_llood_fcn, log_prior_fcn

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
pl_sigma = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32
m = 3 * image_dim * image_dim

batched_posterior = jax.vmap(sample_posterior, in_axes=(0, 0, None, None, None, None, None))


def batch_sample_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns a batch of samples from the posterior distribution at a given temperature."""

    # Sample batch amount of z_posterior samples
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    z_posterior = batched_posterior(
        subkey_batch, 
        x, 
        t, 
        EBM_params, 
        GEN_params, 
        EBM_fwd, 
        GEN_fwd
    )

    return key, z_posterior


def prior_norm(key, EBM_params, EBM_fwd):
    """Returns the normalisation for the prior distribution."""

    key, z = sample_prior(key, EBM_params, EBM_fwd)
    return EBM_fwd(EBM_params, z)


get_priors = jax.vmap(prior_norm, in_axes=(0, None, None))


def llhood(z, x, t, GEN_params, GEN_fwd):
    """Returns the log-likelihood of the generator model for one sample."""

    llhood = log_llood_fcn(z, x, t, GEN_params, GEN_fwd)
    llhood -= jax.scipy.special.logsumexp(stop_gradient(llhood))  # Normalisation is not parameter dependent, required to compare TI and Vanilla
    return llhood.sum()  # log[ p_β(x|z,t) ] = log[ p_β(x1|z1,t) ] + log[ p_β(x2|z2,t) ] + ... + log[ p_β(xN|zN,t) ]


def joint_dist(x, z, t, prior_normaliser, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Returns log[ p_β(x|z,t) ] + log[ p_α(z) ] - log[ Z_{θ,t} ]

    If t=1 AND z ~ p(z|x,t=1), this is the value of the joint logpdf.
    If t>1, this is the value of the posterior logpdf at temperature t for a given z.

    Likelihood normalisation is not paramter dependent, but is dependent on t, so
    we apply logsumexp with stop grad. Prior normalisation is conducted outside of this function.
    """

    prior = log_prior_fcn(z, EBM_params, EBM_fwd) - prior_normaliser
    prior -= jax.scipy.special.logsumexp(stop_gradient(prior))  # None-param dependent normalisation
    return llhood(z, x, t, GEN_params, GEN_fwd) + prior.sum()


joint_logpdf = jax.vmap(joint_dist, in_axes=(0, 0, None, None, None, None, None, None))


def batched_joint_logpdf(key, x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):

    # Prior normalisation is parameter dependent.
    keybatch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = keybatch[0], keybatch[1:]
    prior_normaliser = get_priors(subkey_batch, EBM_params, EBM_fwd).mean(axis=0)
    
    logpdf = joint_logpdf(
        x, 
        z, 
        t, 
        prior_normaliser, 
        EBM_params, 
        GEN_params, 
        EBM_fwd, 
        GEN_fwd
    )

    return logpdf
