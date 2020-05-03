import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRModel
from .util import observe, ExponentialRandomWalk


"""
************************************************************
SEIR model
************************************************************
"""

def SEIR_dynamics(T, params, x0, obs=None, hosp=None, use_hosp=False, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
    
    beta0, sigma, gamma, rw_scale, drift, \
    det_rate, det_noise_scale, hosp_rate, hosp_noise_scale  = params

    beta = numpyro.sample("beta" + suffix,
                  ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

    # Run ODE
    x = SEIRModel.run(T, x0, (beta, sigma, gamma))
    x = x[1:] # first entry duplicates x0
    numpyro.deterministic("x" + suffix, x)

    latent = x[:,4] # cumulative cases

    # Noisy observations
    y = observe("y" + suffix, x[:,4], det_rate, det_noise_scale, obs = obs)
    if use_hosp:
        z = observe("z" + suffix, x[:,4], hosp_rate, hosp_noise_scale, obs = hosp)
    else:
        z = np.zeros_like(y)
        
    return beta, x, y, z


def SEIR_stochastic(T = 50,
                    N = 1e5,
                    T_future = 0,
                    E_duration_est = 4.0,
                    I_duration_est = 2.0,
                    R0_est = 3.0,
                    beta_shape = 1,
                    sigma_shape = 5,
                    gamma_shape = 5,
                    det_rate_est = 0.3,
                    det_rate_conc = 50,
                    det_noise_scale = 0.15,
                    rw_scale = 1e-1,
                    drift_scale = None,
                    obs = None,
                    use_hosp = False,
                    hosp_rate_est = 0.15,
                    hosp_rate_conc = 30,
                    hosp_noise_scale = 0.15,
                    hosp = None):

    '''
    Stochastic SEIR model. Draws random parameters and runs dynamics.
    '''
    
    # Sample initial number of infected individuals
    I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
    E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
    
    # Sample parameters
    sigma = numpyro.sample("sigma", 
                           dist.Gamma(sigma_shape, sigma_shape * E_duration_est))
    
    gamma = numpyro.sample("gamma", 
                           dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

    beta0 = numpyro.sample("beta0", 
                          dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))
        
    det_rate = numpyro.sample("det_rate", 
                              dist.Beta(det_rate_est * det_rate_conc,
                                        (1-det_rate_est) * det_rate_conc))

    if use_hosp:
        hosp_rate = det_rate * numpyro.sample("hosp_rate", 
                                               dist.Beta(hosp_rate_est * hosp_rate_conc,
                                               (1-hosp_rate_est) * hosp_rate_conc))
    else:
        hosp_rate = 0.0
    
    
    if drift_scale is not None:
        drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
    else:
        drift = 0
        
    
    x0 = SEIRModel.seed(N, I0, E0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    if use_hosp:
        hosp0, hosp = (None, None) if hosp is None else (hosp[0], hosp[1:])
        
    
    # First observation
    y0 = observe("y0", x0[4], det_rate, det_noise_scale, obs=obs0)
    if use_hosp:
        z0 = observe("z0", x0[4], hosp_rate, hosp_noise_scale, obs=hosp0)
    else:
        z0 = 0.
        
    params = (beta0, sigma, gamma, 
              rw_scale, drift, 
              det_rate, det_noise_scale, 
              hosp_rate, hosp_noise_scale)
    
    beta, x, y, z = SEIR_dynamics(T, params, x0, 
                                  use_hosp = use_hosp,
                                  obs = obs, 
                                  hosp = hosp)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    z = np.append(z0, z)
    
    if T_future > 0:
        
        params = (beta[-1], sigma, gamma, 
                  rw_scale, drift, 
                  det_rate, det_noise_scale, 
                  hosp_rate, hosp_noise_scale)
        
        beta_f, x_f, y_f, z_f = SEIR_dynamics(T_future+1, params, x[-1,:], 
                                              use_hosp = use_hosp,
                                              suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        z = np.append(z, z_f)
        
    return beta, x, y, z, det_rate, hosp_rate

