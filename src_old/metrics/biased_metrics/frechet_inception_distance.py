import jax.numpy as jnp
from jax.scipy.linalg import cholesky


def calculate_fid(real_features, fake_features, eps=1e-4):
    """
    Frechet Inception Distance.

    https://doi.org/10.48550/arXiv.1706.08500

    Args:
    - real_features: features of real x extracted by InceptionV3
    - fake_features: features of fake x extracted by InceptionV3
    - eps: manually tuned regularization term to avoid numerical instability

    Returns:
    - fid: the Frechet Inception Distance
    """

    # Mean and covariances
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    var_real = jnp.cov(real_features, rowvar=False) + eps * jnp.eye(real_features.shape[-1])
    var_fake = jnp.cov(fake_features, rowvar=False) + eps * jnp.eye(fake_features.shape[-1])

    # Cholesky decomposition
    L_real = cholesky(var_real, lower=True)
    L_fake = cholesky(var_fake, lower=True)

    diff = mu_real - mu_fake
    covmean, _ = jnp.linalg.eigh(L_real @ L_fake)

    return diff @ diff + jnp.trace(var_real) + jnp.trace(var_fake) - 2 * jnp.sum(covmean)


def cosine_similarity(real_features, fake_features):

    # Calculate the norm
    real_norm = jnp.linalg.norm(real_features, axis=1, keepdims=True)
    fake_norm = jnp.linalg.norm(fake_features, axis=1, keepdims=True)

    # Calculate cosine similarity
    cosine_sim = jnp.dot(real_features, fake_features.T) / (real_norm * fake_norm.T)

    cosine_sim = jnp.min(cosine_sim, axis=-1)

    return cosine_sim.mean() 


def calculate_mifid(train_features, val_features, fake_features):
    """
    Memorisation-information FID.

    https://doi.org/10.48550/arXiv.1911.07023
    """

    # Calculate FID
    fid = calculate_fid(val_features, fake_features)

    # Penalise memorisation of training data
    cosine_sim = cosine_similarity(train_features, fake_features)

    # Calculate MIFID
    mifid = fid / (cosine_sim)

    return mifid
