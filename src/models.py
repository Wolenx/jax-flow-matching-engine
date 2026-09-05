import equinox as eqx
import jax.numpy as jnp
import jax

class NeuralVelocityField(eqx.Module):
    """A neural network to guess the velocity vector at time t and location x. It takes in in_size dimension locataion vectors 
    plus one dimension for time and returns the velocity vector of same spacial dimension"""
    mlp: eqx.nn.MLP

    def __init__(self, in_size: int, out_size: int, width: int, depth: int, key):
        # We initialize a standard multi-layer perceptron.
        # Notice: input size is (in_size + 1) to account for time t!
        self.mlp = eqx.nn.MLP(
            in_size=in_size + 1,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=key
        )

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray: #makes the object a callable without using mlp and a model of (x,t)
        
        vec=jnp.concatenate([x,jnp.array([t])])

        return self.mlp(vec)

        pass

