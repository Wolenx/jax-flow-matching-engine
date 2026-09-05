import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from src.mathlib import logliklyhood_estimator
import time
import os


from src.config import IN_SIZE, OUT_SIZE, WIDTH, DEPTH, HEATMAP_RESOLUTION
import src.models as md

# matplotlib.use('Agg')

# 1. Instantiate the skeleton model (random key choice does not matter)
dummy_key = jax.random.key(0)
NeuralField = md.NeuralVelocityField(IN_SIZE, OUT_SIZE, WIDTH, DEPTH, dummy_key)

NeuralField = eqx.tree_deserialise_leaves("./Trained Models/neural_velocity_field_ring.eqx", NeuralField)

print("Model loaded successfully!")

# 1. Define grid resolution (50x50 = 2500 ODE evaluations, fast sequentially)
res = HEATMAP_RESOLUTION
coords = jnp.linspace(-10, 10, res)
X, Y = jnp.meshgrid(coords, coords)
grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1) # (res*res, 2)

# 2. Vectorize your function (this replaces the for-loops entirely)
# It tells JAX: "Take this function that handles 1 point, and make it handle an array of points"
fast_evaluator = jax.jit(jax.vmap(lambda p: logliklyhood_estimator(NeuralField, p)))

# 3. Fire it all at once!
print("Evaluating the entire grid instantly...")

begin=time.time_ns()
log_p_flat = fast_evaluator(grid_points)

# 4. Reshape back into a square and convert log(p) to actual probability
density_map = jnp.exp(log_p_flat).reshape(res, res)
end=time.time_ns()

print("The calculation took : ", (end-begin)/1000000000, "seconds")
# Plot
plt.figure(figsize=(7, 6))
plt.imshow(density_map, extent=[-5, 5, -5, 5], origin="lower", cmap="magma")
plt.colorbar(label="Probability Density $p(x)$")
plt.show()
plt.savefig("heatmap_ring_big.png")
