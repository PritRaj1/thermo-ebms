import jax
from src.MCMC_Samplers.log_pdfs import log_prior_fcn, log_llood_fcn

# log(p(x)) = log(p(x1)) + log(p(x2)) + ... + log(p(xN))
sum_prior = lambda z, EBM_params, EBM_fwd: log_prior_fcn(z, EBM_params, EBM_fwd).sum()
sum_llhood = lambda z, x, t, GEN_params, GEN_fwd: log_llood_fcn(z, x, t, GEN_params, GEN_fwd).sum()

### Grad log probs ###
def prior_grad_log(z, EBM_params, EBM_fwd):
    """Returns ∇_z( log[ p_α(x) ] )"""
    return jax.grad(sum_prior, argnums=0)(z, EBM_params, EBM_fwd)


def posterior_grad_log(z, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns ∇_z( log[p_θ(z | x)] ) = ∇_z( log[ p_β(x | z)^t * p_α(z) ] )"""
    grad_log_llood = jax.grad(sum_llhood, argnums=0)(z, x, t, GEN_params, GEN_fwd)
    grad_prior = prior_grad_log(z, EBM_params, EBM_fwd)
    return grad_log_llood + grad_prior
