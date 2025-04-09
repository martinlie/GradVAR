import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap, value_and_grad
from jax.scipy import stats
from tqdm import tqdm
from gradvar.earlystopping import EarlyStopping

class GradVAR:

      def __init__(self):
            self.A = None # endogenous coefficients
            self.B = None # endogenous bias
            self.C = None # exogenous coefficients
            self.D = None # exogenous bias
            self.p = None # number of lags of endogenous variables
            self.s = None # number of lags of exogenous variables
            self.n = None # number of endogenous vars
            self.m = None # number of exogenous vars

      def _check(self, X, Y):
            if self.A is None or self.B is None or self.p is None or self.n is None:
                  raise ValueError("Model not trained. Run `train()` first.")
            Te, n = X.shape
            Tx, m = Y.shape
            assert(n == self.n)
            assert(m == self.m)
            assert(self.p == self.s) # lag numbers must be equal for now

      def predict(self, X, Y=None):
            if Y is None:
                  Y = jnp.zeros((X.shape[0]+1, 0)) # empty exogenous data

            self._check(X, Y)
            return self._predict(self.A, self.B, self.C, self.D, X, Y)

      def _predict(self, A, B, C, D, X, Y):
            """
            VARX prediction using fully broadcasted and vectorized multiplications.

            Parameters:
            - A: (p, n, n)       — endogenous coefficients
            - B: (n,)            — endogenous bias
            - X: (T, p, n)       — endogenous lags
            - C: (s+1, m, n)     — exogenous coefficients
            - Y: (T, s+1, m)     — exogenous lags
            - D: (n,)            — exogenous bias

            Returns:
            - output: (n,)       — predicted output (collapsed over T)
            """

            # Endogenous effect
            endo_result = jnp.sum(A * X[:, None, :], axis=0)           # shape (p, n, n)
            endogenous_effect = jnp.sum(endo_result, axis=0) + B       # shape (n,)
            
            n = endogenous_effect.shape[0]
            
            # Compute exogenous effect only if included
            def compute_exo(_):
                  exo_result = jnp.sum(C * Y[:, None, :], axis=0)     # shape (s+1, m, n)
                  exogenous_effect = jnp.sum(exo_result, axis=0) + D            # shape (n,)
                  return jnp.resize(exogenous_effect, (n,)) # Ensure padded to (n,)

            def skip_exo(_):
                  return jnp.zeros_like(endogenous_effect)

            include_exogenous = (C.shape[1] > 0) & (C.shape[2] > 0)
            exogenous_effect = lax.cond(include_exogenous, compute_exo, skip_exo, operand=None)

            return endogenous_effect + exogenous_effect

      def _compute_loss_multi(self, A, B, C, D, X, Y, Y_target, W):
            Y_pred = vmap(lambda x,y: self._predict(A, B, C, D, x, y))(X, Y) # multiple
            loss = jnp.mean((Y_target - Y_pred) ** 2) * W
            return loss

      def _prepare_data(self, Ye, Yx, p, s):
            Te, n = Ye.shape # Number of time steps (T) and variables (n)
            Tx, m = Yx.shape
            Xe = jnp.stack([Ye[i:Te - p + i] for i in range(p)], axis=1)         # Shape: (T-p, p, n)
            Xx = jnp.stack([Yx[i:Tx - s + i] for i in range(s+1)], axis=1)       # Shape: (T-p, s+1, m)
            Y_target = Ye[p:] # Shape: (T-p, n)
            return Xe, Xx, Y_target

      def _init_matrices_zeros(self, p, s, n, m):
            A = jnp.zeros((p, n, n))
            B = jnp.zeros((n,))
            C = jnp.zeros((s+1, m, m))
            D = jnp.zeros((m,))
            return A, B, C, D

      def _init_matrices_rng(self, p, s, n, m):
            key_A, key_B, key_C, key_D = jr.split(jr.PRNGKey(0), 4)
            A = jr.normal(key_A, (p, n, n)) * 0.01
            B = jr.normal(key_B, (n,)) * 0.01
            C = jr.normal(key_C, (s+1, m, m)) * 0.01
            D = jr.normal(key_D, (m,)) * 0.01
            return A, B, C, D

      def _init_matrices_glorot(self, p, s, n, m):
            key_A, key_B, key_C, key_D = jr.split(jr.PRNGKey(0), 4)
            A = jr.normal(key_A, (p, n, n)) * jnp.sqrt(2.0 / (n + n))
            B = jr.normal(key_B, (n,)) * jnp.sqrt(2.0 / n)
            C = jr.normal(key_C, (s+1, m, m)) * jnp.sqrt(2.0 / (m + m + 1e-9))
            D = jr.normal(key_D, (m,)) * jnp.sqrt(2.0 / (m + 1e-9))
            return A, B, C, D

      def train(self, Ye, p, Yx=None, num_epochs=1000, learning_rate=0.001, disable_progress=False, A=None, B=None, C=None, D=None, W=1., early_stopping=None, optimizer=None, opt_state=None):
            """ Train VAR model using JAX autodiff """

            if Yx is None:
                  Yx = jnp.zeros((Ye.shape[0], 0)) # empty exogenous data
            s = p # lag numbers must be equal for now
            
            train_losses = jnp.zeros((num_epochs,))
            _, n = Ye.shape # number of endogenous vars
            _, m = Yx.shape # number of exogenous vars
            if A is None or B is None:
                  #A, B, C, D = self._init_matrices_zeros(p, s, n, m)
                  #A, B, C, D = self._init_matrices_rng(p, s, n, m)
                  A, B, C, D = self._init_matrices_glorot(p, s, n, m)

            if optimizer is None:
                  optimizer = optax.adam(learning_rate)
                  opt_state = optimizer.init((A, B, C, D))

            for epoch in tqdm(range(num_epochs), disable=disable_progress):

                  Xe, Xx, Y_target = self._prepare_data(Ye, Yx, p, s)

                  # Autodiff-based gradient calculations and coefficient update
                  loss, grads = value_and_grad(self._compute_loss_multi, argnums=(0, 1, 2, 3))(A, B, C, D, Xe, Xx, Y_target, W)
                  updates, opt_state = optimizer.update(grads, opt_state, (A, B, C, D))
                  A, B, C, D = optax.apply_updates((A, B, C, D), updates)

                  train_losses = train_losses.at[epoch].set(loss)
                  if early_stopping is not None and early_stopping(loss):
                        train_losses = train_losses[:epoch]
                        break

            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.p = p
            self.s = s
            self.n = n
            self.m = m
            return train_losses, optimizer, opt_state

      def forecast(self, Ye, horizon, Yx=None):
            if Yx is None:
                  Yx = jnp.zeros((Ye.shape[0], 0))

            self._check(Ye, Yx)
            Ye_lags = Ye[-self.p:]  # Last p observations as initial input
            Yx_lags = jnp.vstack([ Yx[-self.s:], Yx[-1] ]) # Last s observations + last value TODO Strategy / plan / optimization

            def step(carry, _):
                  Ye_lags, Yx_lags = carry
                  Y_new = self._predict(self.A, self.B, self.C, self.D, Ye_lags, Yx_lags)
                  Ye_lags = jnp.roll(Ye_lags, shift=-1, axis=0).at[-1].set(Y_new)
                  Yx_lags = jnp.roll(Yx_lags, shift=-1, axis=0).at[-1].set(Yx_lags[-1])
                  carry = (Ye_lags, Yx_lags)
                  return carry, Y_new

            _, Y_pred = jax.lax.scan(step, (Ye_lags, Yx_lags), None, length=horizon)
            return Y_pred

      def lagged_forecast(self, Ye, horizon, Yx=None):
            if Yx is None:
                  Yx = jnp.zeros((Ye.shape[0], 0))

            self._check(Ye, Yx)
            T, _ = Ye.shape
            Y_pred = jnp.full((T, self.n), jnp.nan)

            indices = jnp.arange(T - self.p - horizon - 1)
            
            def single_forecast(t):
                  Ye_lags = lax.dynamic_slice(Ye, (t, 0), (self.p, Ye.shape[1]))
                  Yx_lags = lax.dynamic_slice(Yx, (t, 0), (self.s, Yx.shape[1]))
                  return self.forecast(Ye_lags, horizon, Yx_lags)[-1]
            
            Y_new_values = vmap(single_forecast)(indices)
            update_indices = indices + self.p + horizon + 1
            Y_pred = Y_pred.at[update_indices].set(Y_new_values)

            return Y_pred
