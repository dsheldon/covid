# Bayesian compartmental models for COVID-19

This repository contains code for Bayesian estimation of compartmental
models for COVID-19 using [numpyro](https://github.com/pyro-ppl/numpyro) and [jax](https://github.com/google/jax).

## Models

We are experimenting with different Bayesian compartmental models. The basic ingredients are:
* classical [compartmental models from epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
* prior distributions on parameters
* models for time-varying dynamics
* models for partial/noisy observations of confirmed cases and deaths
* Bayesian inference using [numpyro](https://github.com/pyro-ppl/numpyro)

This [Bayesian SEIRD model](docs/Bayesian%20SEIRD%20Model.pdf) is current as of April 27, 2020.

## Team

* [Dan Sheldon](https://people.cs.umass.edu/~sheldon/)
* [Casey gibson](https://gcgibson.github.io/)
* [Nick Reich](https://reichlab.io/people)

## Installation

Our code depends on very recent *development* versions of numpyro and jax. If you don't have these libraries installed or care what versions you get, this should work to install and run our code:
~~~
git clone https://github.com/dsheldon/covid
cd covid
pip install -e .
~~~

## Installation Deatils

If you need to manually install jax and numpyro, here are rough instructions. More details can be found at the project sites.

Install jaxlib (more info [here](https://github.com/google/jax#installation))
~~~
pip install --upgrade jaxlib
~~~

Install jax (more into [here](https://jax.readthedocs.io/en/latest/developer.html#building-from-source))
~~~
git clone https://github.com/google/jax
cd jax
pip install -e .
~~~

Install numpyro (more details [here](https://github.com/pyro-ppl/numpyro))
~~~
git clone https://github.com/pyro-ppl/numpyro.git
cd numpyro
pip install -e .[dev]
~~~
