# thermoEBMs

This package hosts jax/orbax/grain utilities and BF16/FP32 mixed precision training for:

- [latent-space EBMs (Pang et al.)](https://proceedings.neurips.cc/paper_files/paper/2020/file/fa3060edb66e6ff4507886f9912e1ab9-Paper.pdf)
- [Thermodynamic Integration (TI) (Calderhead & Girolami)](https://www.sciencedirect.com/science/article/pii/S0167947309002722)
- [Kolmgorov-Arnold Energy Models (KAEM) (Me)](https://arxiv.org/abs/2506.14167)

## Install

```bash
pip install -e .
pip install -e ".[train,eval,dev,experiment]" # w/ optional deps for orbax/grain training and unbiased fid/kid eval
```

## Experiment

Training:

```python
python main.py model=thermo_kaem training=celeba # single train job
python runner.py # multiple jobs
tensorboard --logdir=runs # launch tensorboard
```

Unbiased image metrics:

```python
python eval.py run runs/thermo_kaem_cifar10 # single trained model
python eval.py all-runs runs/ # for all generated samples in folder
```

Trainer will run distributed data-parallel if a multi-device environment is detected.

## About

The original motivation for this repo, (during my [2023/24 MEng thesis](docs/report.pdf)), was to investigate TI as a means of controlling learning gradient variance in deep generative models. Exploration vs exploitation is still an unanswered question, and previous attempts at investigating it proved unreliable.

<p align="center">
  <img src="https://github.com/PritRaj1/JAX-ThermoEBM/assets/77790119/b526520f-4d92-4eb2-a458-3b0224678a6b" width="50%">
  <br>
  <em>Grid of CelebA face samples with LR gradient variance shown beneath each</em>
</p>

However, the MEng report is old a has some technical inaccuracies!

I've learned a lot since undergrad, and the implementation in this package reflects the updated stats of my newer work KAEM [(summarised here)](https://pritraj1.github.io/portfolio/kaem.html), which offers several major corrections in the learning gradient and TI derivations.

In KAEM, TI and annealing are presented as an interpretable, more parallelisable alternative to diffusion modeling to improve mixing in EBMs, ([explained here](https://pritraj1.github.io/portfolio/defaults.html)).

A [Julia sister repo](https://github.com/thezettascale/KAEM) has also been implemented, which could be faster for single-device training due to its use of [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [EnzymeMLIR](https://github.com/EnzymeAD/enzyme). Enzyme is also available for jax via [Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX), however it's still experimental and not yet suited to deep learning. This repository hosts the multi-device alternative, using grain and data parallelism.
