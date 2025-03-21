import pytest
from gradvar.gradvar import GradVAR
import jax.numpy as jnp
from jax import vmap

def test_static_train():

      av = GradVAR()

      # time series
      Y = jnp.array([
            [1.0, 2.0],  # t=0
            [2.0, 3.0],  # t=1
            [3.0, 4.0],  # t=2
            [4.0, 5.0],  # t=3
            [5.0, 6.0],  # t=4
            [6.0, 7.0],  # t=5
      ])
      p = 2 # lags

      # start coefficients (no randomness)
      A = jnp.array([[[ 0.01622642,  0.02025265],
                      [-0.00433594, -0.00078617]],

                     [[ 0.00176091, -0.00972089],
                      [-0.00495299,  0.00494379]]])
      B = jnp.array([-0.00154437,  0.00084707])

      # test predict before training
      X, Y_target = av._prepare_data(Y, p=2)
      Y_pred = av._predict(A, B, X[0])
      Y_test = jnp.array([0.02856361, 0.03095505])
      assert(jnp.allclose(Y_test, Y_pred, atol=1e-5))

      # train the A and B
      losses = av.train(Y, p, num_epochs=10, learning_rate=0.001, disable_progress=False, A=A, B=B)
      assert(len(losses) == 10)

      testA = jnp.array([[[0.02622642, 0.03025265],
                          [0.00566406, 0.00921383]],

                         [[0.01176091, 0.00027911],
                          [0.00504701, 0.01494379]]])
      testB = jnp.array([0.00845563, 0.01084707])
      assert(jnp.allclose(testA, av.A, atol=1e-5))
      assert(jnp.allclose(testB, av.B, atol=1e-5))
