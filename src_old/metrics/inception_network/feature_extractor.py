from src.metrics.inception_network.inception import InceptionV3
import jax
import jax.numpy as jnp
from jax.image import resize
from functools import partial
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

init_rng = jax.random.PRNGKey(0)
model = InceptionV3(pretrained=True)
params = model.init(init_rng, jnp.ones((1, 256, 256, 3)))
apply_fn = jax.jit(partial(model.apply, train=False))

def inception_pass(images):
    images = resize(images, (256, 256, 3), method="bilinear")
    images = jnp.expand_dims(images, 0)
    features = apply_fn(params, images)
    return features.squeeze()   

fwd_pass = jax.vmap(inception_pass, in_axes=(0))

def scan_pass(_, image_batch):
    return None, fwd_pass(image_batch)
        
def extract_features(images):
    """Extracts features from the InceptionV3 network, (in batches due to memory contraints)"""

    images = jnp.reshape(images, (-1, batch_size, image_dim, image_dim, 3))
    _, features = jax.lax.scan(scan_pass, None, images) 

    return features.reshape(-1, 2048)
