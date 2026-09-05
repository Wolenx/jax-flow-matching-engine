import jax
import jax.numpy as jnp
import numpy as np
import os

def prepare_dataset(raw_filepath: str, output_dir: str, dataset_name: str):
    """Run this once to clean, split, normalize, and save the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load(raw_filepath)
    np.random.shuffle(data) # Ensure random distribution
    
    #80% Train, 20% Test
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    #Compute stats on the TRAINING set
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-7
    
    #Normalize both sets using the training stats
    train_norm = ((train_data - mean) / std).astype(np.float32)
    test_norm = ((test_data - mean) / std).astype(np.float32)
    
    #Save to disk
    np.save(f"{output_dir}/{dataset_name}_train.npy", train_norm)
    np.save(f"{output_dir}/{dataset_name}_test.npy", test_norm)
    np.savez(f"{output_dir}/{dataset_name}_stats.npz", mean=mean, std=std)
    
    print(f"Saved {dataset_name} | Train: {train_norm.shape} | Test: {test_norm.shape}")


def load_dataset_to_vram(filepath: str):
    """
    Loads pre-normalized data and explicitly locks it into GPU VRAM 
    as a pure DeviceArray.
    """
    try:
        data = np.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {filepath}. Run your prep script first.")
    
    # jax.device_put forces the array directly into the GPU memory
    vram_array = jax.device_put(data)
    return vram_array, vram_array.shape[1]

def sample_dataset_vram(key, batch_size, vram_array):
    """
    Slices a batch directly from the pure VRAM array. 
    This function does NOT use closures.
    """
    dataset_size = vram_array.shape[0]
    idx = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=dataset_size)
    return vram_array[idx]

def sample_target_double_spiral(key, num_samples, noise_std=0.03):
    """
    Generates a 2-arm spiral with uniform point density along the arc length.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    half = num_samples // 2
    
    u = jax.random.uniform(k1, shape=(half,), minval=0.1, maxval=1.0)
    theta = jnp.sqrt(u) * 3.5 * jnp.pi  # ~1.75 full turns
    
    r = 0.35 * theta
    
    # Arm 1
    x1 = r * jnp.cos(theta)
    y1 = r * jnp.sin(theta)
    
    # Arm 2
    x2 = -x1
    y2 = -y1
    
    points = jnp.concatenate([
        jnp.stack([x1, y1], axis=-1),
        jnp.stack([x2, y2], axis=-1)
    ], axis=0)
    
    noise = jax.random.normal(k2, shape=points.shape) * noise_std
    points = points + noise 
    return points


def sample_target_ring(key, num_samples, R=3.0):
    """Returns a random sample of data of size num_samples shaped like a ring of inner radius 0.9*R and outer radius of 1.1*R"""  
    key, subkey = jax.random.split(key)
    L_phi = jax.random.uniform(key, shape=(num_samples)) * 2 * jnp.pi
    L_R = jax.random.uniform(subkey, shape=(num_samples), minval=-1., maxval=1.) * R * 0.1 + R
    L_cos = jnp.cos(L_phi)
    L_sin = jnp.sin(L_phi)
    L_trig = jnp.stack([L_cos, L_sin], axis=1)
    return jnp.stack([L_R, L_R], axis=1) * L_trig


def sample_target_hollow_square(key, num_samples, side_length=4.0, noise_std=0.05):
    """
    Generates target points x1 distributed along a hollow square perimeter.
    
    Args:
        key: JAX PRNGKey.
        num_samples: Number of points to sample.
        side_length: Outer edge length of the square.
        noise_std: Standard deviation of Gaussian noise added across the perimeter.
                   (Small non-zero noise prevents infinite density on 1D manifolds).
    """
    k1, k2, k3 = jax.random.split(key, 3)
    half_side = side_length / 2.0

    sides = jax.random.randint(k1, shape=(num_samples,), minval=0, maxval=4)

    u = jax.random.uniform(k2, shape=(num_samples,), minval=-half_side, maxval=half_side)

    # Top (0): (u, half),  Bottom (1): (u, -half),  Left (2): (-half, u),  Right (3): (half, u)
    x = jnp.where(sides < 2, u, jnp.where(sides == 2, -half_side, half_side))
    y = jnp.where(sides == 0, half_side, jnp.where(sides == 1, -half_side, u))

    points = jnp.stack([x, y], axis=-1)

    # 4. Add small Gaussian noise to give the line thickness
    if noise_std > 0:
        noise = jax.random.normal(k3, shape=points.shape) * noise_std
        points = points + noise

    return points

