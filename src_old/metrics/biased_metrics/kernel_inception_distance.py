
import jax.numpy as jnp
import configparser
parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
kernel = str(parser["GENERATION_EVAL"]["KID_KERNEL"])

def rbf_kernel(x, y, gamma):
    """Compute the rbf pairwise kernel matrix."""
    
    # Compute the pairwise squared Euclidean distances
    pairwise_distances_sq = jnp.sum((x[:, None] - y[None, :])**2, axis=-1)
    
    # Compute the kernel matrix
    return jnp.exp(-gamma * pairwise_distances_sq)

def ply_kernel(x, y, gamma, degree=3, coef0=1):
    """Compute the polynomial kernel."""
    
    return (gamma * (x @ y.T) + coef0)**degree

if kernel == 'rbf':
    kernel_matrix = rbf_kernel
elif kernel == 'polynomial':
    kernel_matrix = ply_kernel
else:
    raise ValueError(f"Invalid KID kernel: {kernel}")

def calculate_kid(x, y):
    """
    Calculate the kernel inception distance.
    
    
    https://doi.org/10.48550/arXiv.1801.01401
    """

    # Set the kernel width as 1/num_features
    gamma = 1.0 / x.shape[-1]

    # Compute the RBF kernel matrices
    k_xx = kernel_matrix(x, x, gamma) 
    diag_x = jnp.diag(k_xx)
    k_xx = (k_xx.sum(axis=-1) - diag_x).sum()

    k_yy = kernel_matrix(y, y, gamma) 
    diag_y = jnp.diag(k_yy)
    k_yy = (k_yy.sum(axis=-1) - diag_y).sum()

    k_xy = kernel_matrix(x, y, gamma).sum()

    m = x.shape[0]

    return ((k_xx + k_yy) / (m * (m - 1))) - (2 * k_xy / (m * m))