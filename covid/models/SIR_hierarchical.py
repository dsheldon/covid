import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from ..compartment import SIRModel
from .util import observe, ExponentialRandomWalk


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

