import jax.numpy as jnp
import diffrax
import src.mathlib as mlb

def solve_one_particle_euler(model, x0, dt):
    T = jnp.linspace(0,1,1//dt)
    xt=x0
    dt=T[1]
    for i in range(len(T)):
        v=model(t,xt)
        xt+=dt*v
    return xt


def solve_single_particle_path(model, x0, t_eval=[1.0]):
    """Integrate the path of a single particle. t_eval is the time list of """
    vector_ode = lambda t,x,args : model(t,x)
    term = diffrax.ODETerm(vector_ode)
    solution = diffrax.diffeqsolve(term, diffrax.Tsit5(), t0=0.0, t1=1.0, dt0=0.01, y0=x0, saveat=diffrax.SaveAt(ts=t_eval))
    return solution.ys  # Shape: (50, 2)    
 


def solve_logshift_particle(model, x1, t_eval=[0.0], sample=10):
    """this returns the solution of the augmented vector field integrated. In the end we get the integral from 0 to 1 of the divergeence (thanks to the reversed time axis)
    """

    def augmented_ode_vector_field(model, t, z):
        """Takes in a model, the time t and augmented space vector z = (x(t), delta(log(p(x(t))))) (as a 1D array !) and returns the vector (v(t,x), -div(v(t,x)))"""
        
        x=z[:-1]
        v=model(t,x)
        div=mlb.divergence_router(model,t,x, sample=sample)
        return jnp.concatenate([v,jnp.array([-div])])

    aug_vector_ode = lambda t, y, args : augmented_ode_vector_field(model, t, y)
    term = diffrax.ODETerm(aug_vector_ode)
    y1=jnp.concatenate([x1, jnp.array([0.0])])
    solution = diffrax.diffeqsolve(term, diffrax.Tsit5(), t0=1.0, t1=0.0, dt0=-0.01, y0=y1, saveat=diffrax.SaveAt(ts=t_eval))
    return solution.ys

