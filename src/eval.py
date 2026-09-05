import jax
import jax.numpy as jnp
import equinox as eqx
import src.config as cfg
import src.models as md
import src.mathlib as mlb

def load_model():
    """Deserializes the trained Equinox model dynamically matching the current config."""
    model_filename = f"neural_velocity_field_{cfg.SHAPE}.eqx"
    empty_model = md.NeuralVelocityField(cfg.IN_SIZE, cfg.OUT_SIZE, cfg.WIDTH, cfg.DEPTH, cfg.KEY)
    
    try:
        return eqx.tree_deserialise_leaves(model_filename, empty_model)
    except FileNotFoundError:
        print(f"Error: Could not find {model_filename}. Please train the model first.")
        exit(1)

def evaluate_nll(batch_size: int, vram_array):
    """
    Computes the Negative Log-Likelihood (NLL) across a pure VRAM dataset.
    Chunks the evaluation to prevent Diffrax ODE solver from causing GPU OOM errors.
    """
    if vram_array is None:
        print("Error: No test data provided for evaluation. Ensure --test-data is passed.")
        return

    model = load_model()
    dataset_size = vram_array.shape[0]
    
    # 1. JIT compile the batched log-likelihood estimator
    @jax.jit
    def compute_batch_nll(batch):
        # vmap over the batch dimension, solving the ODE for each point
        batched_log_likelihood = jax.vmap(lambda pt: mlb.logliklyhood_estimator(model, pt, sample=16))(batch)
        
        # We want Negative Log-Likelihood (NLL), so we sum and invert
        return -jnp.sum(batched_log_likelihood)

    total_nll = 0.0
    print(f"Evaluating NLL for {dataset_size} test points in batches of {batch_size}...")

    # 2. Loop over the VRAM array in chunks
    for i in range(0, dataset_size, batch_size):
        batch = vram_array[i : i + batch_size]
        
        # Accumulate the total NLL
        total_nll += compute_batch_nll(batch)
        
        # Simple terminal progress tracker
        if i > 0 and (i // batch_size) % 5 == 0:
            percent = (i / dataset_size) * 100
            print(f"Evaluation Progress: {percent:.1f}%")

    # 3. Compute the final average Test Set NLL
    avg_nll = total_nll / dataset_size
    
    print("========================================")
    print(f"Target: {cfg.SHAPE}")
    print(f"Dimensions: {cfg.IN_SIZE}D")
    print(f"Final Test Set NLL: {avg_nll:.5f}")
    print("========================================")
    
    return avg_nll