# src/viz.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import src.config as cfg
import src.models as md
import src.targets as tg
from src.mathlib import logliklyhood_estimator
import matplotlib.animation as animation

def get_target_sampler():
    """Maps the active SHAPE in config to the correct target sampler."""
    routers = {
        "spiral": tg.sample_target_double_spiral,
        "ring": tg.sample_target_ring,
        "square": tg.sample_target_hollow_square,
    }
    return routers.get(cfg.SHAPE, tg.sample_target_ring) # no visualization for vram data set since they may be of higher dimension

def load_model():
    """Deserializes the trained Equinox model."""
    model_filename = f"Trained Models/neural_velocity_field_{cfg.SHAPE}.eqx"
    empty_model = md.NeuralVelocityField(cfg.IN_SIZE, cfg.OUT_SIZE, cfg.WIDTH, cfg.DEPTH, cfg.KEY)
    try:
        return eqx.tree_deserialise_leaves(model_filename, empty_model)
    except FileNotFoundError:
        print(f"Error: Could not find {model_filename}. Please train the model first.")
        exit(1)

def handle_output(interactive: bool, filename: str):
    """Routes to GUI or disk."""
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved high-res plot to {filename}")
    plt.close()

def compute_density_vram_safe(model, grid_points, batch_size=4096):
    """
    Evaluates log-likelihood in chunks to prevent Diffrax/JAX from causing an Out-Of-Memory (OOM) 
    error on standard GPUs when rendering massive 1000x1000 grids.
    """
    flat_grid = grid_points.reshape(-1, 2)
    num_points = flat_grid.shape[0]
    Z_flat = np.zeros(num_points)

    # JIT compile the batched estimator
    @jax.jit
    def compute_batch(batch):
        return jax.vmap(lambda pt: logliklyhood_estimator(model, pt, sample=16))(batch)

    print(f"Evaluating {num_points} grid points in batches of {batch_size}...")
    for i in range(0, num_points, batch_size):
        batch = flat_grid[i : i + batch_size]
        Z_flat[i : i + batch_size] = compute_batch(batch)
        
        # Simple progress tracker for the terminal
        if i > 0 and (i // batch_size) % 10 == 0:
            print(f"Rendering: {(i / num_points) * 100:.1f}%")

    return Z_flat

def plot_heatmap(interactive: bool):
    """Generates a high-fidelity density map overlaid with ground-truth target points."""
    model = load_model()
    res = cfg.HEATMAP_RESOLUTION
    
    # 1. Create the coordinate grid
    x = jnp.linspace(-4, 4, res)
    y = jnp.linspace(-4, 4, res)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.stack([X, Y], axis=-1)
    
    # 2. Safely compute density
    Z_flat = compute_density_vram_safe(model, grid)
    Z = np.exp(Z_flat.reshape(res, res)) # Convert log-likelihood to standard density
    
    # 3. Sample ground truth points to overlay
    target_sampler = get_target_sampler()
    key = jax.random.key(cfg.SEED)
    true_samples = target_sampler(key, 2000)

    # 4. Render the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Base density map
    c = ax.pcolormesh(X, Y, Z, shading='gouraud', cmap='magma', rasterized=True)
    fig.colorbar(c, ax=ax, label='Learned Probability Density')
    
    # Overlay ground truth as semi-transparent cyan dots
    ax.scatter(true_samples[:, 0], true_samples[:, 1], s=1, color='cyan', alpha=0.4, label='Ground Truth Samples')
    
    ax.set_title(f"Continuous-Time Density Map - {cfg.SHAPE.capitalize()}", fontsize=14, pad=15)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    handle_output(interactive, f"heatmap_{cfg.SHAPE}.png")

def plot_vector_field(interactive: bool):
    """
    Generates an animated quiver plot showing the neural velocity field.
    All vectors are normalized to the same length to prevent visual clutter,
    using color gradients to represent velocity magnitude.
    """
    model = load_model()
    res = cfg.VECTORMAP_RESOLUTION # Default is 50[cite: 1]
    
    x = np.linspace(-4, 4, res)
    y = np.linspace(-4, 4, res)
    X, Y = np.meshgrid(x, y)
    grid = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor='#0f0f16')
    ax.set_facecolor('#0f0f16')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_aspect('equal')
    
    ax.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.tick_params(colors='white')

    @jax.jit
    def get_velocities(t_val):
        return jax.vmap(lambda pt: model(t_val, pt))(grid)

    init_v = get_velocities(0.0)
    U = np.array(init_v[:, 0])
    V = np.array(init_v[:, 1])
    speed = np.sqrt(U**2 + V**2)
    
    # Safely normalize the initial vectors
    speed_safe = np.where(speed < 1e-7, 1e-7, speed)
    U_norm = U / speed_safe
    V_norm = V / speed_safe

    # Scale parameter is adjusted since all arrow lengths are exactly 1
    Q = ax.quiver(X, Y, U_norm, V_norm, speed, cmap='turbo', alpha=0.85, 
                  scale=res * 1.2, width=0.003, pivot='mid')
    
    title = ax.set_title(f"Velocity Field t=0.00 - {cfg.SHAPE.capitalize()}", 
                         color='white', fontsize=14, pad=15)
    
    fig.colorbar(Q, ax=ax, label='Velocity Magnitude').ax.yaxis.label.set_color('white')
    Q.colorbar.ax.tick_params(colors='white')

    def update(frame_t):
        v_t = get_velocities(frame_t)
        u = np.array(v_t[:, 0])
        v = np.array(v_t[:, 1])
        spd = np.sqrt(u**2 + v**2)
        
        # Safely normalize the frame vectors
        spd_safe = np.where(spd < 1e-7, 1e-7, spd)
        u_norm = u / spd_safe
        v_norm = v / spd_safe
        
        # Update positions with normalized vectors, but keep true speed for color
        Q.set_UVC(u_norm, v_norm, spd)
        title.set_text(f"Velocity Field t={frame_t:.2f} - {cfg.SHAPE.capitalize()}")
        return Q, title

    frames = np.linspace(0.0, 1.0, 40)
    
    print("Generating frames... (this takes a moment to compile on the first frame)")
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    if interactive:
        plt.show()
    else:
        filename = f"vector_field_{cfg.SHAPE}.gif"
        print(f"Rendering animation to {filename}...")
        anim.save(filename, writer='pillow', fps=15)
        print(f"Saved normalized fluid animation to {filename}")
        
    plt.close()
