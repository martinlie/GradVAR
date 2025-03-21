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

VAR models need continous time series - when modeling, ensure that there are no gaps in the time series data.

## Why this lib?

* dis-continouse time series
* optimize matrices by gradient decent
* update training with new data
* custom loss function and weighting profiles

## Pre-requisites

* stationary data
* standardized/z-score normalized data

## Why Use JAX?

* Autodiff (grad): No need to manually derive gradients.
* JIT Compilation (jit): Fast execution.
* GPU/TPU Support: Scalable for large data.