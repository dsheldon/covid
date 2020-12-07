from setuptools import find_packages, setup

setup(
    name='covid',
    version="0.0.1",
    description='Bayesian COVID-19 models',
    packages=find_packages(include=['covid', 'covid.*']),
    url='https://github.com/dsheldon/covid-19',
    author='Dan Sheldon',
    author_email='sheldon@cs.umass.edu',
    install_requires=[
        'jaxlib>=0.1.45',
        'patsy>=0.5.1',
        'jax==0.2.3',
        'numpyro @ git+https://github.com/pyro-ppl/numpyro.git#egg=numpyro'
    ],
    keywords=' machine learning bayesian statistics',
    license='MIT'
)
