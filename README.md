# Bayesian compartmental models for COVID-19

This repository contains code for Bayesian estimation of compartmental
models for COVID-19.

## Models

We are experimenting with different models. This [Bayesian SEIR model](Bayesian%20SEIR%20Model.pdf) is fairly current as of April 2, 2020.

## Installation

Uses numpyro and jax. Requires most recent *development* versions of both
projects from their github repos:

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

Clone this repo and install:
~~~
git clone https://github.com/dsheldon/covid
cd covid
pip install -e .
~~~
