from setuptools import setup, find_packages

setup(
    name="GradVAR",
    version="2025.3.1",
    description="Gradient update Vector Autoregression modeling library",
    author="Martin Forsberg Lie",
    packages=find_packages(),
    install_requires=[
      "jax",
      "optax",
      "tqdm"
    ]
)
