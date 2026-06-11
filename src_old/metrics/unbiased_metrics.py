import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from src.metrics.inception_network.feature_extractor import extract_features
from src.pipeline.generate import generate_images
from src.metrics.biased_metrics.frechet_inception_distance import (
    calculate_fid,
    calculate_mifid,
)
from src.metrics.biased_metrics.kernel_inception_distance import calculate_kid
from src.metrics.biased_metrics.inception_score import calculate_is


def get_metrics(train_x_features, val_x_features, x_pred_features):

    fid = calculate_fid(val_x_features, x_pred_features)
    mifid = calculate_mifid(train_x_features, val_x_features, x_pred_features)
    kid = calculate_kid(val_x_features, x_pred_features)
    incep = calculate_is(x_pred_features)

    return fid, mifid, kid, incep


def metrics_fcn(
    key, sample_size, train_activations, val_activations, x_pred_activations
):

    # Build random subsets
    key, subkey = jax.random.split(key)
    train_act_i = jax.random.choice(
        subkey,
        train_activations,
        shape=(sample_size,),
        replace=True,
    )
    val_act_i = jax.random.choice(
        subkey,
        val_activations,
        shape=(sample_size,),
        replace=True,
    )
    x_pred_act_i = jax.random.choice(
        subkey,
        x_pred_activations,
        shape=(sample_size,),
        replace=True,
    )

    
    fid, mifid, kid, incep = get_metrics(train_act_i, val_act_i, x_pred_act_i)

    return key, (fid, mifid, kid, incep)


def profile_generation(
    key,
    params_tup,
    train_x,
    val_x,
    fwd_fcn_tup,
    min_samples,
    max_samples,
    num_points,
    num_plot,
):
    """
    Function to profile the generative capacity of the model.
    Traditional image-quality metrics are generally biased,
    so following the previous work of Chong et. al, we aim to
    remove bias by interpolating to an infinite sample size.

    https://doi.org/10.48550/arXiv.1911.07023

    Args:
    - key: PRNG key
    - x: all test data
    - gen_fcn: generative function, partially preloaded with immutables
    - min_samples: minimum number of samples
    - max_samples: maximum number of samples
    - num_points: number of points to profile
    - num_plot: number of images to plot for visual inspection

    Returns:
    - fid_inf: the unbiased FID
    - mifid_inf: the unbiased MIFID
    - kid_inf: the unbiased KID
    - four_real: 4 random real images
    - four_fake: 4 random fake images
    """

    # Preload the generation and profile functions with immutable arguments
    gen_fcn = partial(
        generate_images,
        params_tup=params_tup,
        num_images=max_samples,
        fwd_fcn_tup=fwd_fcn_tup,
    )

    if max_samples > len(val_x):
        raise ValueError("Max samples cannot exceed the number of test images.")
    elif min_samples > max_samples:
        raise ValueError("Min samples cannot exceed max samples.")

    # Generate the maximum number of samples
    key, x_pred = gen_fcn(key)

    # Get all activations

    train_activations = extract_features(train_x)
    val_activations = extract_features(val_x)
    x_pred_activations = extract_features(x_pred)
    loaded_metrics = partial(
        metrics_fcn,
        train_activations=train_activations,
        val_activations=val_activations,
        x_pred_activations=x_pred_activations,
    )

    fid = jnp.zeros(num_points)
    mifid = jnp.zeros(num_points)
    kid = jnp.zeros(num_points)
    incep = jnp.zeros(num_points)

    # Compute image metrics for each batch size
    batch_sizes = np.linspace(min_samples, max_samples, num_points).astype(int)
    for idx, sample_size in enumerate(batch_sizes):
        key, (fid_i, mifid_i, kid_i, is_i) = loaded_metrics(key, sample_size)
        fid = fid.at[idx].set(fid_i)
        mifid = mifid.at[idx].set(mifid_i)
        kid = kid.at[idx].set(kid_i)
        incep = incep.at[idx].set(is_i)

    # Intepolate to infinite sample size by fitting a line and taking the intercept at 1/N = 0
    _, fid_inf = jnp.polyfit(1 / batch_sizes, fid, 1)
    _, mifid_inf = jnp.polyfit(1 / batch_sizes, mifid, 1)
    _, kid_inf = jnp.polyfit(1 / batch_sizes, kid, 1)
    _, incep_inf = jnp.polyfit(1 / batch_sizes, incep, 1)

    # Return 4 random images for plotting (visual inspection)
    key, subkey = jax.random.split(key)
    real = jax.random.choice(subkey, val_x, shape=(num_plot,), replace=False)
    fake = jax.random.choice(subkey, x_pred, shape=(num_plot,), replace=False)

    return key, fid_inf, mifid_inf, kid_inf, incep_inf, real, fake
