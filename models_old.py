import jax
import jax.numpy as np
from jax.experimental.ode import odeint

import numpyro
import numpyro.distributions as dist

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from compartment import SIRModel, SEIRModel



def _observe_beta_binom(name, latent, det_rate, det_conc, obs=None):
    '''
    Make observations of a latent variable using BetaBinomial.
    
    (Cannot get inference to work with this model. Sigh.)
    '''
    mask = True
    
    if obs is not None:
        mask = np.isfinite(obs) & (obs >= 0) & (obs <= latent)
        obs = np.where(mask, obs, 0.0)
         
    det_rate = np.broadcast_to(det_rate, latent.shape)
    
    latent = np.ceil(latent).astype('int32') # ensure integer
    
    d = dist.BetaBinomial(det_conc * det_rate, det_conc * (1-det_rate), latent)
    
    with numpyro.handlers.mask(mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y


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


"""
************************************************************
SIR hierarchical
************************************************************
"""

def SIR_dynamics_hierarchical(SIR, T, params, x0, obs = None, suffix=""):

    '''
    Run SIR dynamics for T time steps
    '''
    
    beta0, gamma, det_rate, det_conc, rw_scale = params

    # Add a dimension to these for broadcasting with 2d arrays (num_places x T)
    beta0 = beta0[:,None]
    det_rate = det_rate[:,None]
    
    with numpyro.plate("num_places", beta0.shape[0]):
        beta = numpyro.sample("beta" + suffix, 
                      ExponentialRandomWalk(loc = beta0, scale=rw_scale, num_steps=T-1))
    

    # Run ODE
    apply_model = lambda x0, beta, gamma: SIR.run(T, x0, (beta, gamma))
    x = jax.vmap(apply_model)(x0, beta, gamma)

    # TODO: workaround for vmap bug
    #x = np.stack([apply_model(xx, b, g) for xx, b, g in zip(x0, beta, gamma)])
    #x = SIR.run_batch(T, x0, (beta, gamma))
    
    x = x[:,1:,:] # drop first time step from result (duplicates initial value)
    numpyro.deterministic("x" + suffix, x)
   
    # Noisy observations
    y = observe("y" + suffix, x[:,:,3], det_rate, det_conc, obs = obs)

    return beta, x, y


def SIR_hierarchical(num_places = 1,
                     T = 50, 
                     N = 1e5,
                     T_future = 0,
                     duration_mean = 10,
                     R0_mean = 2.2,
                     gamma_shape = 5,
                     beta_shape = 5,
                     det_rate_mean = 0.3,
                     det_rate_conc = 50,
                     det_conc = 100,
                     rw_scale = 5e-2,
                     obs = None):
    '''
    Hierarchical SIR model
    '''


    '''
    Draw shared parameters
    '''
    
    gamma_ = numpyro.sample("gamma_", 
                     dist.Gamma(gamma_shape, 
                                gamma_shape * duration_mean))

    beta_ = numpyro.sample("beta_", 
                             dist.Gamma(beta_shape, 
                                 beta_shape * duration_mean/R0_mean))

    det_rate_ = numpyro.sample("det_rate_", 
                               dist.Beta(det_rate_mean * det_rate_conc,
                                         (1 - det_rate_mean) * det_rate_conc))
    
        
    # Broadcast to correct size
    N = np.broadcast_to(N, (num_places,))
    

    '''
    Draw place-specific parameters
    '''
    with numpyro.plate("num_places", num_places):
        
        I0 = numpyro.sample("I0", dist.Uniform(0, N*0.02))
                
        gamma = numpyro.sample("gamma", dist.Gamma(20, 20 / gamma_))
        
        beta0 = numpyro.sample("beta0", dist.Gamma(20, 20 / beta_))
        
        det_rate = numpyro.sample("det_rate", dist.Beta(100*det_rate_, 100*(1-det_rate_)))

    '''
    Run model for each place
    '''
    SIR = SIRModel()
    x0 = jax.vmap(SIR.seed)(N, I0)
    numpyro.deterministic("x0", x0)

    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[:,0], obs[:,1:])
    
    # First observation
    y0 = observe("y0", x0[:,3], det_rate, det_conc, obs=obs0)
        
    # Run dynamics
    params = (beta0, gamma, det_rate, det_conc, rw_scale)
    beta, x, y = SIR_dynamics_hierarchical(SIR, T, params, x0, obs = obs)
    
    x = np.concatenate((x0[:,None,:], x), axis=1)
    y = np.concatenate((y0[:,None], y), axis=1)
    
    if T_future > 0:
        
        params = (beta[:,-1], gamma, det_rate, det_conc, rw_scale)
        
        beta_f, x_f, y_f = SIR_dynamics_hierarchical(SIR, 
                                                     T_future+1, 
                                                     params, x
                                                     [:,-1,:], 
                                                     suffix="_future")
        
        x = np.concatenate((x, x_f), axis=1)
        y = np.concatenate((y, y_f), axis=1)
        
    return beta, x, y, det_rate

