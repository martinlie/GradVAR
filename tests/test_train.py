import pytest
from gradvar.gradvar import GradVAR
import jax.numpy as jnp
from jax import vmap

def test_static_train_with_exo():

      av = GradVAR()

      # time series
      Ye = jnp.array([
            [1.0, 2.0],  # t=0
            [2.0, 3.0],  # t=1
            [3.0, 4.0],  # t=2
            [4.0, 5.0],  # t=3
            [5.0, 6.0],  # t=4
            [6.0, 7.0],  # t=5
      ])
      p = 2 # lags

      Yx = jnp.array([
            [3.0],  # t=0
            [4.0],  # t=1
            [5.0],  # t=2
            [6.0],  # t=3
            [7.0],  # t=4
            [8.0],  # t=5
      ])
      s = 2

      # start coefficients (no randomness)
      A = jnp.array([[[ 0.01622642,  0.02025265],
                      [-0.00433594, -0.00078617]],

                     [[ 0.00176091, -0.00972089],
                      [-0.00495299,  0.00494379]]])
      B = jnp.array([-0.00154437,  0.00084707])
      C = jnp.array([[[ 0.01622642],
                      [-0.00433594]],

                     [[ 0.00176091],
                      [-0.00495299]],
                      
                      [[ 0.00176091],
                      [-0.00495299]]])
      D = jnp.array([-0.00154437,  0.00084707])

      # test predict before training
      Xe, Xx, Y_target = av._prepare_data(Ye, Yx, p, s)
      Y_pred = av._predict(A, B, C, D, Xe[0], Xx[0]) # first sample
      Y_test = jnp.array([7.1e-05, 2.2e-02])

      assert(Y_pred.shape == Y_test.shape)
      assert(jnp.allclose(Y_test, Y_pred, atol=1e-3))

      # continue training the coefficients
      losses, *_ = av.train(Ye, p, Yx, num_epochs=10, learning_rate=0.001, disable_progress=False, A=A, B=B, C=C, D=D)
      assert(len(losses) == 10)
      assert(av.A.shape == (2, 2, 2))
      assert(av.B.shape == (2,))

def test_static_train_without_exo():

      av = GradVAR()

      # time series
      Ye = jnp.array([
            [1.0, 2.0],  # t=0
            [2.0, 3.0],  # t=1
            [3.0, 4.0],  # t=2
            [4.0, 5.0],  # t=3
            [5.0, 6.0],  # t=4
            [6.0, 7.0],  # t=5
      ])
      p = 2 # lags

      Yx = jnp.zeros((Ye.shape[0], 0)) # <--- even though there are no exogenous, we must shape it to have time series shape
      s = 2

      # start coefficients (no randomness)
      A = jnp.array([[[ 0.01622642,  0.02025265],
                      [-0.00433594, -0.00078617]],

                     [[ 0.00176091, -0.00972089],
                      [-0.00495299,  0.00494379]]])
      B = jnp.array([-0.00154437,  0.00084707])

      C = jnp.zeros((s+1, 0, 0))
      D = jnp.zeros((0,))

      # test predict before training
      Xe, Xx, Y_target = av._prepare_data(Ye, Yx, p, s)
      Y_pred = av._predict(A, B, C, D, Xe[0], Xx[0]) # first sample
      Y_test = jnp.array([-0.005, 0.015])

      assert(Y_pred.shape == Y_test.shape)
      assert(jnp.allclose(Y_test, Y_pred, atol=1e-3))

      # continue training the coefficients
      losses, *_ = av.train(Ye, p, Yx, num_epochs=10, learning_rate=0.001, disable_progress=False, A=A, B=B, C=C, D=D)
      assert(len(losses) == 10)

      assert(av.A.shape == (2, 2, 2))
      assert(av.B.shape == (2,))

def test_train():

      av = GradVAR()

      # time series
      Ye = jnp.array([
            [1.0, 2.0],  # t=0
            [2.0, 3.0],  # t=1
            [3.0, 4.0],  # t=2
            [4.0, 5.0],  # t=3
            [5.0, 6.0],  # t=4
            [6.0, 7.0],  # t=5
      ])
      p = 2 # lags

      Yx = jnp.array([
            [3.0],  # t=0
            [4.0],  # t=1
            [5.0],  # t=2
            [6.0],  # t=3
            [7.0],  # t=4
            [8.0],  # t=5
      ])

      losses, *_ = av.train(Ye, p, Yx, num_epochs=10, learning_rate=0.001, disable_progress=True)
      assert(len(losses) == 10)