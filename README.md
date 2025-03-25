# GradVAR

Gradient update Vector Autoregression modeling library

## Modeling with Vector Autoregression

Vector Autoregression (VAR) is a statistical model used to capture the linear interdependencies among multiple time series data. It generalizes the univariate autoregressive model to multiple time series, allowing each variable to be modeled as a linear function of its own past values and the past values of other variables in the system.

A VAR model with $k$ variables and $p$ lags can be expressed as a system of equations:

$$
Y_t = \mu + \sum_{i=1}^{p} A_i Y_{t-i} + \epsilon_t
$$

Where:
* $Y_t$ is the vector of endogenous variables at time $t$
* $\mu$ is a vector of constants (intercepts),
* $A_i$ is a matrix of coefficients for the $i$-th lag,
* $p$ is the number of lags,
* $\epsilon_t$ is a vector of error terms (shocks or innovations) at time $t$

## Why this lib?

Traditional Vector Autoregression (VAR) models require continuous time series and are often estimated using least squares methods, which can be inefficient for large datasets or evolving time series. GradVAR introduces a more flexible and adaptive approach by leveraging gradient-based optimization techniques.

* **Discontinuous Time Series** Unlike standard VAR models, GradVAR can be used to process time series with missing observations or irregular sampling
* **Gradient-Based Matrix Optimization** Instead of relying on closed-form solutions, GradVAR optimizes the coefficient matrices using gradient descent, allowing for better performance on large datasets.
* **Incremental Training with New Data** GradVAR allows continuous learning by updating the existing model with new observations.
* **Custom Loss Functions and Weighting** You can define your own loss function and assign weights to different time periods or variables, tailoring the model to specific forecasting needs.

## Pre-requisites

The library works on simple data matrices, data frame functionality is out of scope for this library and should be implemented elsewhere.

* **Stationary Data** Ensure the time series is stationary; differencing or transformation may be needed.
* **Standardized Data** Use z-score normalization for consistent scaling across variables before calling the library, or use batch normalization externally

## Why Use JAX?

JAX is a powerful framework that combines NumPy-like syntax with automatic differentiation and just-in-time (JIT) compilation and GPU/TPU support. Further, the gradient update is done by the Adam optimizer.

# Installation

## Install via pip

The library can be installed directly from the repository using:

      pip install gradvar@git+https://github.com/martinlie/gradvar.git@main#egg=gradvar

into the current environment. 

## Install the Library Locally

Use `pip install -e` to create an **editable installation**, by first cloning the repository and then:

      pip install -e <path to gradvar>
    
This creates a symbolic link to your library directory, so changes in the library are immediately reflected in any project that uses it.

## Usage

Refer to the examples folder for notebooks that demonstrate how to use the library.

# Development

## Build the library

```
pip install -r requirements-dev.txt
python -m build
```

## Release WHL file

```
conda install gh --channel conda-forge
gh auth login
gh release create release-0.1.0 dist/gradvar-0.1.0-py3-none-any.whl --generate-notes
```
