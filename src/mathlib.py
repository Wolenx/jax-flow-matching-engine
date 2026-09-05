import jax.numpy as jnp
import jax
from src.solvers import solve_logshift_particle
import equinox as eqx

def divergence(F, t, x):
    """Takes in a Function F(t,x) of time and space and returns the divergence at x at time t"""
    F_t=lambda x: F(t, x)
    A=jax.jacrev(F_t)
    return jnp.trace(A(x))

def divergenceByHandHutchinson(F, t, x, key=jnp.array([0, 2026], dtype=jnp.uint32), sample=1000): 
    """Divergence calculator optimized. Takes in a Function F(t,x) of time and space and returns the divergence at x at time t
    Uses Skilling-Hutchinson to ESTIMATE the divergence"""

    time_hash = jnp.int32(t * 100000)
    step_key = jax.random.fold_in(key, time_hash)

    d=x.shape[0]
    epsilon=jax.random.rademacher(step_key, shape=(sample, d), dtype=jnp.float32)
    F_t=lambda x_in: F(t, x_in)
    _, vjp_F=jax.vjp(F_t, x)

    batched_vjp=jax.vmap(lambda v : vjp_F(v)[0])
    v_J= batched_vjp(epsilon)

    estimations = jnp.sum(v_J * epsilon, axis=-1)
    es=jnp.mean(estimations)

    return es

def divHutchinson(F, t, x, key=jnp.array([0, 2026], dtype=jnp.uint32), sample=512): 
    """Divergence calculator optimized. Takes in a Function F(t,x) of time and space and returns the divergence at x at time t
    Uses Skilling-Hutchinson to ESTIMATE the divergence"""

    d=x.shape[0]
    epsilon=jax.random.normal(key, shape=(sample, d))
    F_t=lambda x_in: F(t, x_in)
    _, vjp_F=jax.vjp(F_t, x)
    def singel_estimation(v):
        return jnp.dot(vjp_F(v)[0],v)
    
    s=jax.vmap(singel_estimation)(epsilon)
    es=jnp.mean(s)

    return es

@eqx.filter_jit
def divergence_router(F, t, x, key=jax.random.key(2026), threshold=20, sample=16):
    """
    Routes to Exact Trace for D <= threshold, Hutchinson for D > threshold.
    Because x.shape is static, this 'if' statement disappears during compilation!
    """
    dim = x.shape[0]
    
    # This evaluates at COMPILE time. 
    # The GPU kernel will only contain the code for the chosen branch.
    if dim <= threshold:
        return divergence(F, t, x)
    else:
        return divHutchinson(F, t, x, key, sample)

@jax.jit
def compute_ot_paires_good(x0_batch, x1_batch, num_iters=20):
    """This function takes in the 2 batches and returns a new batch x1 where the points are rearrenged to get assigne each point of x1 to the closest of x0.
    This version uses a Soft Transport (Sinkhorn) so the batch is a little modified."""
    
    dtype = x0_batch.dtype
    stabilizer = jnp.where(dtype == jnp.float64, 1e-16, 1e-8)

    size=x0_batch.shape[0] # (N,d)

    x0_c=jnp.sum(x0_batch*x0_batch, axis=1, keepdims=True)
    x1_c=jnp.sum(x1_batch*x1_batch, axis=1, keepdims=True) # the last flag makes the vector keep shape (N,1) instead of (N)

    C=x0_c + x1_c.T - 2.0*jnp.dot(x0_batch, x1_batch.T)
    C=jnp.maximum(C, 0.0)
    
    epsilon=jnp.maximum(jnp.median(C)*0.02, 0.001) # epsilon is adapted to 2% of the median to have it big enough and still get convergence
    K=jnp.exp((jnp.min(C)-C)/epsilon)

    u = jnp.ones((size,), dtype=dtype) / size
    v = jnp.ones((size,), dtype=dtype) / size

    def step(i, state):
        u, v = state
        u = 1.0 / (size*(jnp.dot(K, v) + stabilizer))
        v = 1.0 / (size*(jnp.dot(K.T, u) + stabilizer))
        return u, v
    
    u, v = jax.lax.fori_loop(0, num_iters, step, (u, v))

    P = u[:, None] * K * v[None, :]
    x1_matched = jnp.dot(P, x1_batch)*size

    return x1_matched

def logliklyhood_estimator(model, x, sample=16):
    """gives the log likelyhood of a point x in space"""
    sol=solve_logshift_particle(model, x, sample=sample)[0]
    x0=sol[:-1]
    delta=sol[-1]
    loginit=-len(x)*0.5*jnp.log(2*jnp.pi)-0.5*jnp.dot(x0,x0)
    return loginit-delta #delta here refers to the integral from 0 to 1 of the divergence
