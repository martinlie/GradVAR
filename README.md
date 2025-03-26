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

Furthermore, by ensuring the model remains comprehensible and explainable, we can gain deeper insights into the underlying system. The final result is two matrices A and B that can forecast a system with autoregressive properties and can be easily implemented in a system with only basic arithmetic. Here is a conceptual example:

```python
# Example usage with 2 variables (k=2) and 2 lags (p=2)
A = jnp.array([
    [[0.6, -0.3], [0.2, 0.5]],   # A1 (lag 1)
    [[0.1,  0.2], [-0.4, 0.3]],  # A2 (lag 2)
])
B = jnp.array([0.5, -0.2])  # Bias term
Y_lags = jnp.array([
    [10, 5],   # Y_{t-1}
    [7, 6],    # Y_{t-2}
])

# Step 1: Initialize result array
result = jnp.zeros((k,))

# Step 2: Element-wise multiplication and summation over i and k
for i in range(A.shape[0]):  # Iterate over the first dimension (p)
    for k in range(A.shape[2]):  # Iterate over the third dimension (k)
        result += A[i, :, k] * X[i, k]  # Element-wise multiplication and accumulation

# Step 3: Add the bias B
output = result + B
```

## Pre-requisites

The library works on simple data matrices, data frame functionality is out of scope for this library and should be implemented elsewhere.

* **Stationary Data** Ensure the time series is stationary; differencing or transformation may be needed.
* **Standardized Data** Use z-score normalization for consistent scaling across variables before calling the library, or use batch normalization externally

## Why Use JAX?

JAX is a powerful framework that combines NumPy-like syntax with automatic differentiation and just-in-time (JIT) compilation and GPU/TPU support. The gradient update is done by the Adam optimizer.

## Planned enhancements

**Statistical summary**: Residual correlation analysis, compute optimal lag number, p-value calculation for each variable and lag

## Install via pip

The library can be installed directly from the repository using:

```
pip install gradvar
```

into the current environment.

## Usage

Refer to the examples folder for notebooks that demonstrate how to use the library.
