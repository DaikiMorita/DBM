from setuptools import setup, find_packages
import sys

sys.path.append('./DBM/test')

setup(
    name="Deep Boltzmann Machine",
    version="0.1",
    description="Deep Boltzmann Machine, referring a paper called"
                "'Efficient Learning of Deep Boltzmann Machines' by Salakhutdinov and Larochelle in 2010",
    author="Daiki Morita",
    url="https://github.com/DaikiMorita/RBM",
    packages=find_packages(),
    install_requires=['numpy', 'tqdm', 'requests'],
    dependency_links="",
    test_suite='DBM.test.suite.suite'
)
