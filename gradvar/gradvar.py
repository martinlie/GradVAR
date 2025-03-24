import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap, value_and_grad
from jax.scipy import stats
from tqdm import tqdm
from gradvar.earlystopping import EarlyStopping

class GradVAR:

      def __init__(self):
            self.A = None # lag coefficient matrices
            self.B = None # bias term
            self.p = None # lags
            self.k = None # num of vars

      def _check(self, D):
            if self.A is None or self.B is None or self.p is None or self.k is None:
                  raise ValueError("Model not trained. Run `train()` first.")
            T, k = D.shape
            assert(k == self.k)

      def predict(self, X):
            self._check(X)
            return self._predict(self.A, self.B, X)

      def _predict(self, A, B, X):
            # Einsum decomposition (matrix multiplication)
            result = jnp.sum(A * X[:, None, :], axis=0)
            output = jnp.sum(result, axis=0) + B
            return output

      def _compute_loss_multi(self, A, B, X, Y_target):
            # could use A_normalized = batch_norm(A)  # Batch normalization
            # TODO loss weighting
            Y_pred = vmap(lambda x: self._predict(A, B, x))(X) # multiple
            loss = jnp.mean((Y_target - Y_pred) ** 2)
            return loss

      def _prepare_data(self, Y, p):
            T, k = Y.shape # Number of time steps (T) and variables (k)
            X = jnp.stack([Y[i:T - p + i] for i in range(p)], axis=1)  # Shape: (T-p, p, k)
            Y_target = Y[p:] # Shape: (T-p, k)
            return X, Y_target

      def _init_matrices(self, p, k):
            key_A, key_B = jr.split(jr.PRNGKey(0))
            #A = jr.normal(key_A, (p, k, k)) * 0.01  # VAR Coefficients
            #B = jr.normal(key_B, (k,)) * 0.01  # Intercept term
            A = jnp.zeros((p, k, k))  # VAR Coefficients
            B = jnp.zeros((k,))       # Intercept term
            #A = jr.normal(key_A, (p, k, k)) * jnp.sqrt(2.0 / (k + k))  # VAR Coefficients (Glorot)
            #B = jr.normal(key_B, (k,)) * jnp.sqrt(2.0 / k)             # Intercept term (Glorot)
            return A, B

      def train(self, Y, p, num_epochs=1000, learning_rate=0.001, disable_progress=False, A=None, B=None, early_stopping=None):
            """ Train VAR model using JAX autodiff """
            
            train_losses = jnp.zeros((num_epochs,))
            _, k = Y.shape # k=num vars
            if A is None or B is None:
                  A, B = self._init_matrices(p, k)

            optimizer = optax.adam(learning_rate)
            opt_state = optimizer.init((A, B))

            for epoch in tqdm(range(num_epochs), disable=disable_progress):

                  X, Y_target = self._prepare_data(Y, p=p) # TODO pick some batches

                  # Autodiff-based gradient calculations and coefficient update
                  loss, grads = value_and_grad(self._compute_loss_multi, argnums=(0, 1))(A, B, X, Y_target)
                  updates, opt_state = optimizer.update(grads, opt_state, (A, B))
                  A, B = optax.apply_updates((A, B), updates)

                  train_losses = train_losses.at[epoch].set(loss)
                  if early_stopping is not None and early_stopping(loss):
                        train_losses = train_losses[:epoch]
                        break

            self.A = A
            self.B = B
            self.p = p
            self.k = k
            return train_losses

      def forecast(self, Y, horizon):
            self._check(Y)
            Y_pred = jnp.zeros((horizon, self.k))
            Y_lags = Y[-self.p:]  # Last p observations as initial input

            for t in range(horizon):
                  Y_new = self._predict(self.A, self.B, Y_lags)
                  Y_lags = jnp.roll(Y_lags, shift=-1, axis=0).at[-1].set(Y_new)
                  Y_pred = Y_pred.at[t].set(Y_new)

            return Y_pred

      def lagged_forecast(self, Y, horizon, disable_progress=False):
            self._check(Y)
            T, _ = Y.shape
            Y_pred = jnp.full((T, self.k), jnp.nan)          

            for t in tqdm(range(T - self.p - horizon - 1), disable=disable_progress):      
                  Y_lags = Y[t:t + self.p]
                  Y_new = self.forecast(Y_lags, horizon)[-1]
                  Y_pred = Y_pred.at[t+self.p+horizon+1].set(Y_new)
            return Y_pred
