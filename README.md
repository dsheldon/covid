# Bayesian compartmental models for COVID-19

Uses numpyro and jax. Requires most recent *development* versions of both
projects from their github repos:

Install jaxlib (more info [here](https://github.com/google/jax#installation))
~~~
pip install --upgrade jax jaxlib  # CPU-only version
~~~

Install jax (more into [here](https://jax.readthedocs.io/en/latest/developer.html#building-from-source))
~~~
git clone https://github.com/google/jax
cd jax
pip install -e .
~~~

Install numpyro (more details [here](https://github.com/pyro-ppl/numpyro)
~~~
git clone https://github.com/pyro-ppl/numpyro.git
cd numpyro
pip install -e .[dev]
~~~
