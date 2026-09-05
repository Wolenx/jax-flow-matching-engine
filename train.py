import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import src.models as md
import src.targets as tg
import src.losses as ls
import src.config as cfg

def get_target_sampler():
    def wrap(math_sampler):
        return lambda k, b_size, _vram: math_sampler(k, b_size)
        
    routers = {
        "double_spiral": wrap(tg.sample_target_double_spiral),
        "ring": wrap(tg.sample_target_ring),
        "square": wrap(tg.sample_target_hollow_square),
        "personalized": tg.sample_dataset_vram 
    }
    return routers.get(cfg.SHAPE, wrap(tg.sample_target_ring))

def train(epochs: int, batch_size: int, vram_array=None):
    NeuralField = md.NeuralVelocityField(cfg.IN_SIZE, cfg.OUT_SIZE, cfg.WIDTH, cfg.DEPTH, cfg.KEY)
    
    base_key = jax.random.key(cfg.SEED)
    _, loop_key = jax.random.split(base_key)
    
    optimizer = optax.adam(learning_rate=1e-3)
    params = eqx.filter(NeuralField, eqx.is_array)
    opt_state = optimizer.init(params)
    
    target_sampler = get_target_sampler()
    
    @eqx.filter_jit
    def make_step(model, opt_state, optimizer, x0_batch, x1_batch, subk):
        loss_val, grads = eqx.filter_value_and_grad(ls.loss_fn)(model, x0_batch, x1_batch, subk)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val

    @eqx.filter_jit
    def train_block(model, opt_state, k, steps_per_block, vram_data):
        dynamic_model, static_model = eqx.partition(model, eqx.is_array)
        
        def scan_step(carry, _):
            dyn_model, opt, curr_k = carry
            full_model = eqx.combine(dyn_model, static_model)
            
            curr_k, subk_x0, subk_x1, subk_loss = jax.random.split(curr_k, 4)
            
            x0_batch = jax.random.normal(subk_x0, shape=(batch_size, cfg.IN_SIZE))
            x1_batch = target_sampler(subk_x1, batch_size, vram_data)
            
            full_model, opt, loss = make_step(full_model, opt, optimizer, x0_batch, x1_batch, subk_loss)
            new_dyn_model, _ = eqx.partition(full_model, eqx.is_array)
            
            return (new_dyn_model, opt, curr_k), loss

        (final_dyn_model, new_opt_state, new_k), losses = jax.lax.scan(scan_step, (dynamic_model, opt_state, k), xs=None, length=steps_per_block)
        return eqx.combine(final_dyn_model, static_model), new_opt_state, new_k, jnp.mean(losses)

    steps_per_block = 128
    num_blocks = epochs // steps_per_block
    
    for block in range(num_blocks):
        NeuralField, opt_state, loop_key, avg_loss = train_block(NeuralField, opt_state, loop_key, steps_per_block, vram_array)
        
        if block % (max(1, num_blocks // 10)) == 0:
            percent = ((block + 1) * steps_per_block / epochs) * 100
            print(f"Progress: {percent:.1f}% | Avg Loss: {avg_loss:.5f}")

    model_filename = f"neural_velocity_field_{cfg.SHAPE}.eqx"
    eqx.tree_serialise_leaves(model_filename, NeuralField)
    print(f"Model saved to {model_filename}")