import jax
import configparser

from src.loss_computation.loss_helper_fcns import batched_joint_logpdf, batch_sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

def vanilla_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns - log[ p(x|θ) ] = - E_{z|x,t=1}[ log[ p_α(z) ] + log[ p_β(x|z,t=1) ] ]"""

    key, z_posterior = batch_sample_posterior(key, x, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
    key, subkey = jax.random.split(key)
    llhood = batched_joint_logpdf(subkey, x, z_posterior, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd).mean()
    return -llhood, key