from setuptools import setup

REQUIRED = [
    "matplotlib",
    "pyaml",
    "optuna==3.4.0",
]


setup(name="dhexp", packages=["dhexp"], install_requires=REQUIRED)
