import os
import sys
from setuptools import find_packages, setup

try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to read README.md:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''

setup(
    name='covid',
    version="0.0.1",
    description='Bayesian COVID-19 models',
    packages=find_packages(include=['covid', 'covid.*']),
    url='https://github.com/dsheldon/covid-19',
    author='Dan Sheldon',
    author_email='sheldon@cs.umass.edu',
    #install_requires=[
    # TODO: jaxlib, jax, numpyro, patsy
    #],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=' machine learning bayesian statistics',
    license='MIT'
)
