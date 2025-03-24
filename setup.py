from setuptools import setup, find_packages

setup(
    name="GradVAR",
    version="0.1.0",
    description="Gradient update Vector Autoregression modeling library",
    author="Martin Lie",
    packages=find_packages(),
    install_requires=[
      "jax",
      "optax",
      "tqdm"
    ]
)
