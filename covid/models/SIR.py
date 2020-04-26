import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from ..compartment import SIRModel
from .util import observe, ExponentialRandomWalk


"""
************************************************************
SIR model
************************************************************
"""

def SIR_dynamics(T, params, x0, obs = None, suffix=""):

    '''
    Run SIR dynamics for T time steps
    '''
    
    beta0, gamma, det_rate, det_conc, drift_scale = params

    beta = numpyro.sample("beta" + suffix, 
                  ExponentialRandomWalk(loc = beta0, scale=drift_scale, num_steps=T-1))

    # Run ODE
    x = SIRModel.run(T, x0, (beta, gamma))
    x = x[1:,:] # drop first time step from result (duplicates initial value)
    numpyro.deterministic("x" + suffix, x)
   
    # Noisy observations
    y = observe("y" + suffix, x[:,3], det_rate, det_conc, obs = obs)

    return beta, x, y


def SIR_stochastic(T = 50, 
                   N = 1e5,
                   T_future = 0,
                   duration_mean = 10,
                   R0_mean = 2.2,
                   gamma_shape = 5,
                   beta_shape = 5,
                   det_rate_mean = 0.3,
                   det_rate_conc = 50,
                   det_conc = 100,
                   drift_scale = 5e-2,
                   obs = None):

    '''
    Stochastic SIR model. Draws random parameters and runs dynamics.
    '''

    # Sample initial number of infected individuals
    I0 = numpyro.sample("I0", dist.Uniform(0, N*0.02))
    
    # Sample parameters
    gamma = numpyro.sample("gamma", 
                           dist.Gamma(gamma_shape, gamma_shape * duration_mean))
        
    beta0 = numpyro.sample("beta0", 
                          dist.Gamma(beta_shape, beta_shape * duration_mean/R0_mean))

        
    det_rate = numpyro.sample("det_rate", 
                              dist.Beta(det_rate_mean * det_rate_conc,
                                        (1-det_rate_mean) * det_rate_conc))
    

    x0 = SIRModel.seed(N, I0)
    numpyro.deterministic("x0", x0)

    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    
    # First observation
    y0 = observe("y0", x0[3], det_rate, det_conc, obs=obs0)
    
    # Run dynamics
    params = (beta0, gamma, det_rate, det_conc, drift_scale)
    
    beta, x, y = SIR_dynamics(T, params, x0, obs = obs)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    
    if T_future > 0:
        
        params = (beta[-1], gamma, det_rate, det_conc, drift_scale)
        
        beta_f, x_f, y_f = SIR_dynamics(T_future+1, params, x[-1,:], suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        
    return beta, x, y, det_rate
