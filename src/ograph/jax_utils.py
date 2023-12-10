import jax.numpy as jnp


def safe_get(arr: jnp.ndarray, idx):
    return arr.at[idx].get(mode="fill", fill_value=jnp.nan)
