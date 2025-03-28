import pytest
import jax.numpy as jnp
import jax.random as jr
from gradvar.gradvar import GradVAR
from gradvar.earlystopping import EarlyStopping

def generate_sinusoidal_data(T, noise_std=0.05, key=jr.PRNGKey(0)):
    """ Generate a 2D time series: one sine and one cosine wave with noise """
    t1 = jnp.linspace(-jnp.pi * 8, jnp.pi * 8, T)
    t2 = jnp.linspace(-jnp.pi * 2, jnp.pi * 2, T)
    t3 = jnp.linspace(-jnp.pi * 4, jnp.pi * 4, T)
    noise = noise_std * jr.normal(key, (3, T))
    Y = jnp.stack([jnp.sin(t1)*0.5, jnp.cos(t2), jnp.sin(t3) + jnp.sin(t3*1.5)]) + noise
    return Y.T

@pytest.fixture
def test_earlystopping():
    key = jr.PRNGKey(42)
    Y = generate_sinusoidal_data(T = 400, noise_std=0.05, key=key)

    av = GradVAR()
    early_stopping = EarlyStopping(patience=30, min_delta=1e-5)
    num_epochs=10000
    losses, *_ = av.train(Y, p=50, learning_rate=0.001, num_epochs=num_epochs, early_stopping = early_stopping)

    assert(len(losses) < num_epochs)

    return av, Y

def test_forecast(test_earlystopping):
    av, Y = test_earlystopping

    Y_forecast = av.forecast(Y, 250)

    assert(Y_forecast.shape[0] == 250)
    assert(Y_forecast.shape[1] == 3)

def test_lagged_forecast(test_earlystopping):
    av, Y = test_earlystopping

    horizon = 25
    Y_forecast = av.lagged_forecast(Y, horizon)

    assert(Y_forecast.shape == Y.shape)