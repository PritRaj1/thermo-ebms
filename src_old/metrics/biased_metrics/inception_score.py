import jax
import jax.numpy as jnp
from optax import kl_divergence as kl_div
from flax.linen import softmax

KL_div = lambda x: jnp.sum(x * (jnp.log(x) - jnp.log(jnp.mean(x, axis=0))))
vmapped_div = jax.vmap(KL_div)

@jax.jit
def calculate_is(fake_features, splits=10):
    """exp(E_{x ~ p(x)}[KL(p(y|x) || p(y))])"""

    prob = softmax(fake_features, axis=-1)
    split_indices = [i * (prob.shape[0] // splits) for i in range(splits + 1)]
    parts = jnp.array_split(prob, split_indices)
    div_stack = vmapped_div(jnp.array(parts[1:-1]))

    return jnp.exp(jnp.mean(div_stack, axis=0))


