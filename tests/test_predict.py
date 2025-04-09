import pytest
from gradvar.gradvar import GradVAR
import jax.numpy as jnp
from jax import vmap

def test_single_predict():
      av = GradVAR()

      # Example usage with 2 variables (k=2) and 2 lags (p=2)
      av.p = 2
      av.s = 2
      av.n = 2
      av.m = 0

      av.A = jnp.array([
            [[0.6, -0.3], [0.2, 0.5]],  # A1 (lag 1)
            [[0.1,  0.2], [-0.4, 0.3]]  # A2 (lag 2)
      ])

      av.B = jnp.array([0.5, -0.2])  # Bias term
      av.C = jnp.zeros((av.s+1, 0, 0))
      av.D = jnp.zeros((0,))

      Xe = jnp.array([
            [10, 5],  # Y_{t-1}
            [7, 6]    # Y_{t-2}
      ])

      Y_test = jnp.array([6.4, 3.8])
      Y_pred = av.predict(Xe)

      assert(Y_pred.shape == Y_test.shape)
      assert(jnp.allclose(Y_test, Y_pred, atol=1e-1))


def test_multiple_predict():
      av = GradVAR()

      av.p = 2
      av.s = 2
      av.n = 2
      av.m = 0

      av.A = jnp.array([
            [[0.6, -0.3], [0.2, 0.5]],  # A1 (lag 1)
            [[0.1,  0.2], [-0.4, 0.3]],  # A2 (lag 2)
      ])
      av.B = jnp.array([0.5, -0.2])  # Bias term
      av.C = jnp.zeros((av.s+1, 0, 0))
      av.D = jnp.zeros((0,))

      Ye = jnp.array([
            [1.0, 10.0],  # t=0
            [2.0, 20.0],  # t=1
            [3.0, 30.0],  # t=2
            [4.0, 40.0],  # t=3
            [5.0, 50.0],  # t=4
            [6.0, 60.0],  # t=5
      ])
      Yx = jnp.zeros((Ye.shape[0], 0))

      Y_test = jnp.array(
            [[ 0.7, 11.8],
             [ 1.2, 18.8],
             [ 1.7, 25.8],
             [ 2.2, 32.8]])

      Xe, Xx, Y_target = av._prepare_data(Ye, Yx, p=2, s=2) # one batch

      Y_pred = vmap(lambda x: av.predict(x))(Xe)

      assert(Y_pred.shape == Y_test.shape)
      assert(jnp.allclose(Y_test, Y_pred, atol=1e-1))
