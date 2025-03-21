import pytest
from gradvar.gradvar import GradVAR
import jax.numpy as jnp
from jax import vmap

def test_single_predict():
      av = GradVAR()

      # Example usage with 2 variables (k=2) and 2 lags (p=2)
      av.A = jnp.array([
            [[0.6, -0.3], [0.2, 0.5]],  # A1 (lag 1)
            [[0.1,  0.2], [-0.4, 0.3]]  # A2 (lag 2)
      ])

      av.B = jnp.array([0.5, -0.2])  # Bias term
      av.p = 2

      X = jnp.array([
            [10, 5],  # Y_{t-1}
            [7, 6]    # Y_{t-2}
      ])

      Y_test = jnp.array([10.4,  9.7])
      
      Y_pred = av.predict(X)

      assert(jnp.array_equal(Y_test, Y_pred))


def test_multiple_predict():
      av = GradVAR()

      av.A = jnp.array([
            [[0.6, -0.3], [0.2, 0.5]],  # A1 (lag 1)
            [[0.1,  0.2], [-0.4, 0.3]],  # A2 (lag 2)
      ])

      av.B = jnp.array([0.5, -0.2])  # Bias term
      av.p = 2

      Y = jnp.array([
            [1.0, 10.0],  # t=0
            [2.0, 20.0],  # t=1
            [3.0, 30.0],  # t=2
            [4.0, 40.0],  # t=3
            [5.0, 50.0],  # t=4
            [6.0, 60.0],  # t=5
      ])

      Y_test = jnp.array(
            [[12.7, 12. ],
             [20.2, 19.5],
             [27.7, 27. ],
             [35.2, 34.5]])

      X, Y_target = av._prepare_data(Y, p=2) # one batch

      Y_pred = vmap(lambda x: av.predict(x))(X)

      assert(jnp.array_equal(Y_test, Y_pred))
