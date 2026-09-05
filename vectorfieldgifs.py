import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.config import IN_SIZE, OUT_SIZE, WIDTH, DEPTH
import src.models as md

matplotlib.use('Agg')

# 1. Instantiate the skeleton model (random key choice does not matter)
dummy_key = jax.random.key(0)
NeuralField = md.NeuralVelocityField(IN_SIZE, OUT_SIZE, WIDTH, DEPTH, dummy_key)

NeuralField = eqx.tree_deserialise_leaves("./Trained Models/neural_velocity_field_lattice.eqx", NeuralField)

print("Model loaded successfully!")

# 1. Set up spatial evaluation grid
res = 25
extent = 4.0
x_coords = jnp.linspace(-extent, extent, res)
y_coords = jnp.linspace(-extent, extent, res)
X, Y = jnp.meshgrid(x_coords, y_coords)
grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

# 2. JIT-compiled vector field evaluator
@eqx.filter_jit
def evaluate_grid_velocities(model, t, points):
    return jax.vmap(lambda x: model(t, x))(points)

# 3. Initial frame setup
fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

vel_0 = np.asarray(evaluate_grid_velocities(NeuralField, 0.0, grid_points))
U_0 = vel_0[:, 0].reshape(res, res)
V_0 = vel_0[:, 1].reshape(res, res)
speeds_0 = np.sqrt(vel_0[:, 0]**2 + vel_0[:, 1]**2)  # 1D array (625,)

# Initial quiver object
q = ax.quiver(X, Y, U_0, V_0, speeds_0, cmap="coolwarm", scale=35, pivot="mid", width=0.005)

ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect("equal")
plt.colorbar(q, ax=ax, label=r"Velocity Magnitude $\|v_\theta(t, x)\|$")

# 4. Frame update function
num_frames = 60
time_steps = jnp.linspace(0.0, 1.0, num_frames)

def update(frame):
    t = float(time_steps[frame])
    
    # Evaluate velocities and cast to NumPy arrays
    vel = np.asarray(evaluate_grid_velocities(NeuralField, t, grid_points))
    U = vel[:, 0].reshape(res, res)
    V = vel[:, 1].reshape(res, res)
    speeds_1d = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)  # Must be 1D for set_array

    # Update vector components and color intensity
    q.set_UVC(U, V)
    q.set_array(speeds_1d)
    ax.set_title(rf"Learned Velocity Field $v_\theta(t, x)$ | $t = {t:.2f}$")
    return (q,)

# 5. Render and save GIF
print("Rendering animation frames...")
anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
anim.save("vector_field_lattice.gif", writer=PillowWriter(fps=20))
plt.close(fig)

print("Saved gif successfully!")