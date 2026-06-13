# thermoEBMs

Latent-space energy-based models (EBMs) for unsupervised generative modeling, with Thermodynamic Integration to improve EBM mixing and preserve single-pass, interpretable inference post-training.

This package hosts jax/orbax/grain utilities for [latent-space EBMs](https://proceedings.neurips.cc/paper_files/paper/2020/file/fa3060edb66e6ff4507886f9912e1ab9-Paper.pdf), [Thermodynamic Integration (TI)](https://www.sciencedirect.com/science/article/pii/S0167947309002722), (and soon [Kolmgorov-Arnold Energy Models (KAEM)](https://arxiv.org/abs/2506.14167)).

## Install

```bash
pip install -e .
pip install -e ".[train,eval,dev]" # w/ optional deps for orbax/grain training and unbiased fid/kid eval
```

## About

The original motivation for this work, (during my 2023/24 [MEng thesis](docs/Shaping_the_Learning_Gradient_Distribution_with_Thermodynamic_Integration.pdf)), was to investigate TI as a means of controlling learning gradient variance in deep generative models. Exploration vs exploitation is still an unanswered question, and previous attempts at investigating it proved unreliable.

<p align="center">
  <img src="https://github.com/PritRaj1/JAX-ThermoEBM/assets/77790119/b526520f-4d92-4eb2-a458-3b0224678a6b" width="50%">
  <br>
  <em>Grid of CelebA face samples with LR gradient variance shown beneath each</em>
</p>

However, the MEng report is old a has some technical inaccuracies! I've learned a lot since undergrad, and the implementation in this package reflects the updated stats of my newer work [KAEM](https://pritraj1.github.io/kaem.html), which offer several major corrections in the learning gradient and TI derivations. In KAEM, TI and annealing are presented as an interpretable, more parallelisable alternative to diffusion modeling to improve mixing in EBMs, ([explained here](https://pritraj1.github.io/defaults.html)).

A [Julia sister repo](https://github.com/thezettascale/KAEM) has also been implemented, which may be faster for single-gpu training due to its use of [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [EnzymeMLIR](https://github.com/EnzymeAD/enzyme). Enzyme is also available for jax via [Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX), however it's still experimental and not yet suited to deep learning. I've written a [summary](https://pritraj1.github.io/kaem.html#implementation) of the differences between jax and julia, jax/grain are preferable for distributed training.
