from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from numpy.polynomial.legendre import leggauss

from ..config import KAEMConfig


def kernel(
    z: jax.Array,
    translation: jax.Array,
    bandwidth: jax.Array,
    tau: jax.Array,
) -> jax.Array:
    z_scaled = (z - translation) / bandwidth
    real = jnp.cos(tau * z_scaled) - jnp.exp(-(tau**2) / 2)
    envelope = jnp.exp(-(z_scaled**2) / 2)
    return real * envelope


def expand_z(x: np.ndarray) -> jax.Array:
    return jnp.expand_dims(jnp.array(x), axis=(1, 2, 3))


def vmap_component(function: Callable, x: jax.Array) -> jax.Array:
    return jax.vmap(function)(jnp.expand_dims(x, axis=1)).squeeze(axis=2)


class KAN(nnx.Module):
    """1D latent density function"""

    init_domain: tuple[float, float] = (-3.0, 3.0)

    def __init__(self, config: KAEMConfig, P: int, rngs: nnx.Rngs):
        self.mixture = config.mixture
        self.sigma = config.p0_stddev

        # Kolmogorov-Arnold Theorem width choices, n -> 2n+1
        self.Q = (P - 1) // 2 if self.mixture else 2 * P + 1
        self.P = P

        self.translation = nnx.Param(rngs.normal((1, 1, self.Q, P)))
        self.bandwidth = nnx.Param(rngs.normal((1, 1, self.Q, P)))
        self.tau = nnx.Param(rngs.normal((1, 1, self.Q, P)))

        # Mixture component to sample
        self.reg = config.mixture_regularization
        self.alpha = nnx.Param(jnp.ones((1, 1, self.Q, P))) if self.mixture else None
        self.component = (
            nnx.Variable(jnp.arange(self.Q)[None, None, :, None])
            if self.mixture
            else None
        )

        # Gauss–Legendre quadrature for Inverse Transform
        self.numquad = config.numquad
        lo = jnp.full((1, 1, self.Q, self.P), -self.init_domain[0])
        hi = jnp.full((1, 1, self.Q, self.P), self.init_domain[1])
        nodes, weights = self.adapt_gauss(lo, hi)
        self.nodes = nnx.Variable(nodes)
        self.weights = nnx.Variable(weights)

        # Domain adaption
        self.update_every = config.domain_update_freq
        self.adaption_converage = config.domain_update_coverage

    def adapt_gauss(self, lo: jax.Array, hi: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Adapt Gauss-Legendre integration domain"""
        nodes, weights = leggauss(self.numquad)
        nodes, weights = jnp.array(nodes), jnp.array(weights)
        nodes, weights = expand_z(nodes), expand_z(weights)

        nodes = 0.5 * (hi - lo) * nodes + 0.5 * (lo + hi)
        weights = weights * 0.5 * (hi - lo)
        return nodes, weights

    def domain_update(self, z: jax.Array) -> None:
        z_sorted = jnp.sort(z, axis=0)
        N = z_sorted.shape[0]
        n_cov = int(self.adaption_converage * N)

        intervals_low = z_sorted[: N - n_cov, :, :, :]
        intervals_high = z_sorted[n_cov:, :, :, :]
        widths = intervals_high - intervals_low

        best_idx = jnp.argmin(widths, axis=0, keepdims=True)
        lo = jnp.take_along_axis(intervals_low, best_idx, axis=0)
        hi = jnp.take_along_axis(intervals_high, best_idx, axis=0)

        nodes, weights = self.adapt_gauss(lo, hi)
        self.nodes[...] = nodes
        self.weights[...] = weights

    def log_p0(self, z: jax.Array) -> jax.Array:
        """π_0(z) = N(0, 1), in_shape = (N_quad, Q, P)"""
        return (
            -0.5 * (z / self.sigma) ** 2
            - jnp.log(self.sigma)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )

    def sample_mixture(self, key: jax.Array, N: int) -> jax.Array:
        """Sample uniformly from Categorical(1:mixture_components). Called outside JIT"""
        if self.mixture:
            key, subkey = jax.random.split(key)
            self.component.set_value(
                jax.random.categorical(
                    subkey,
                    logits=self.alpha,
                    axis=-2,
                    shape=(N, 1, 1, self.P),
                )
            )

        return key

    def select_component(self, x: jax.Array) -> jax.Array:
        """Choose mixture component along Q dim"""
        if not self.mixture:
            return x

        return jnp.take_along_axis(x, self.component, axis=-2)

    def __call__(
        self,
        z: jax.Array,
    ) -> jax.Array:
        return kernel(z, self.translation, self.bandwidth, self.tau)

    def en(self, z: jax.Array) -> jax.Array:
        f = self(z)
        if not self.mixture:
            return f.sum()

        f = f + nnx.log_softmax(self.alpha, axis=-2)
        return nnx.logsumexp(f, axis=-2).sum()

    def prior_score(self, z: jax.Array) -> jax.Array:
        grad_f = jax.grad(self.en)(z)
        return grad_f - z / (self.sigma**2)

    def componentwise_f(self, z: jax.Array) -> jax.Array:
        """
        In: (numsamples, 1, Q, P)
        Out: (num_quad, numsamples, 1, P) if mixture else (numquad, numsamples, Q, P))
        """
        translation = self.select_component(self.translation)
        bandwidth = self.select_component(self.bandwidth)
        tau = self.select_component(self.tau)
        z = self.select_component(z)
        return kernel(z, translation, bandwidth, tau)

    def loss(self, z_post: jax.Array, z_prior: jax.Array) -> jax.Array:
        """Constrastive divergence: E_{p_θ(z | x)}[f(z)] - E_{p_α(z)}[f(z)]"""
        reg = 0
        if self.mixture:
            reg = self.reg * jnp.sum(jnp.abs(self.alpha))

        return -self.en(z_post) + self.en(z_prior) + reg * z_prior.shape[0]

    def pdf_per_node(self):
        """Returns normalized pdf per density"""
        f = vmap_component(self.componentwise_f, self.nodes)
        z = vmap_component(self.select_component, self.nodes)
        weights = vmap_component(self.select_component, self.weights)
        return weights * jnp.exp(f + self.log_p0(z)), z
