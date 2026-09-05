import jax.numpy as jnp
import jax


def loss_fn(model, x0_batch, x1_batch, key):
    t = jax.random.uniform(key, shape=(x0_batch.shape[0],1))
    
    Xt = (1 - t) * x0_batch + t * x1_batch
    Ut = x1_batch - x0_batch

    Vpred = jax.vmap(model)(jnp.squeeze(t,axis=1), Xt)
    Diff = Vpred - Ut
    meanerror = jnp.mean(jnp.square(Diff))

    return meanerror

