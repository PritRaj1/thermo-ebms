import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from functools import partial
from src.MCMC_Samplers.sample_distributions import sample_prior
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sigma = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32

def generate(key, _, params_tup, fwd_fcn_tup):
    """Generate a batch image from the generator."""

    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_gen = fwd_fcn_tup[1](params_tup[1], stop_gradient(z))

    return key, x_gen

batch_generate = jax.vmap(generate, in_axes=(0, None, None, None))

@partial(jax.jit, static_argnums=(2, 3))
def generate_images(key, params_tup, num_images, fwd_fcn_tup):
    """Generates 'num_images' images from the generator."""

    loaded_gen = partial(generate, params_tup=params_tup, fwd_fcn_tup=fwd_fcn_tup)
    batch_generate = jax.vmap(loaded_gen, in_axes=0)

    # The generator struggles with memory, so generate in batches
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    _, images = jax.lax.scan(
        f=batch_generate,
        init=subkey_batch, 
        xs=None, 
        length=num_images // batch_size
    )
    images = jnp.reshape(images, (-1, image_dim, image_dim, 3))

    return key, images
