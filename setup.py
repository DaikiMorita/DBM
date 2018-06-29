from setuptools import setup, find_packages
import sys

sys.path.append('.DBM/src')
sys.path.append('.DBM/test')

setup(
    name="Deep Boltzmann Machine",
    version="0.1",
    packages=find_packages(),
    test_suite='DBM.test.suite.suite'
)
