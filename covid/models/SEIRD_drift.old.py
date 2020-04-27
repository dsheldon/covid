import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, ExponentialRandomWalk


"""
************************************************************
SEIRD model
************************************************************
"""

def SEIR_dynamics(T, params, x0, obs=None, death=None, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
    
    beta0, sigma, gamma, rw_scale, drift, \
    det_prob, det_noise_scale, death_prob, death_rate, det_prob_d  = params
    
    beta = numpyro.sample("beta" + suffix,
                  ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift= drift, num_steps=T-1))

    # Run ODE
    x = SEIRModel.run(T, x0, (beta, sigma, gamma, death_prob, death_rate))
    x = x[1:] # first entry duplicates x0
    numpyro.deterministic("x" + suffix, x)


    # Noisy observations
    with numpyro.handlers.scale(scale_factor=0.5):
        y = observe("y" + suffix, x[:,6], det_prob, det_noise_scale, obs = obs)
        
    with numpyro.handlers.scale(scale_factor=2.0):
        z = observe("z" + suffix, x[:,5], det_prob_d, det_noise_scale, obs = death)
        
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
                    det_prob_est = 0.3,
                    det_prob_conc = 50,
                    det_noise_scale = 0.15,
                    rw_scale = 1e-1,
                    drift_scale = None,
                    obs = None,
                    death=None,
                    death_prob_est = 0.15,
                    death_prob_cont = 30,
                    death_noise_scale = 0.15,
                    hosp = None):

    '''
    Stochastic SEIR model. Draws random parameters and runs dynamics.
    '''
    
    # Sample initial number of infected individuals
    I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
    E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
    H0 = numpyro.sample("H0", dist.Uniform(0, 0.02*N))
    D0 = numpyro.sample("D0", dist.Uniform(0, 0.02*N))
    
    # Sample parameters
    sigma = numpyro.sample("sigma", 
                           dist.Gamma(sigma_shape, sigma_shape * E_duration_est))
    
    gamma = numpyro.sample("gamma", 
                           dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

    beta0 = numpyro.sample("beta0", 
                           dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))
        
    det_prob = numpyro.sample("det_prob", 
                              dist.Beta(det_prob_est * det_prob_conc,
                                        (1-det_prob_est) * det_prob_conc))
    
    det_prob_d = numpyro.sample("det_prob_d", 
                                dist.Beta(.9 * 100,
                                          (1-.9) * 100))
    
    death_prob = numpyro.sample("death_prob", 
                                dist.Beta(.1 * 100,
                                          (1-.1) * 100))
    
    death_rate = numpyro.sample("death_rate", 
                                dist.Beta(.1 * 100,
                                          (1-.1) * 100))
    
    if drift_scale is not None:
        drift = numpyro.sample("drift", dist.Normal(loc=np.log(.5), scale=.01))
        drift = -np.exp(drift)
    else:
        drift = 0
        
    
    x0 = SEIRModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    death0, death = (None, None) if death is None else (death[0], death[1:])

    
    # First observation
    with numpyro.handlers.scale(scale_factor=0.5):
        y0 = observe("y0", x0[6], det_prob, det_noise_scale, obs=obs0)
    z0 = observe("z0", x0[5], det_prob_d, det_noise_scale, obs=death0)

        
    params = (beta0, sigma, gamma, 
              rw_scale, drift, 
              det_prob, det_noise_scale, 
              death_prob, death_rate, det_prob_d)
    
    beta, x, y, z = SEIR_dynamics(T, params, x0, 
                                  obs = obs, 
                                  death = death)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    z = np.append(z0, z)
    
    if T_future > 0:
        
        params = (beta[-1], sigma, gamma, 
                  rw_scale, drift, 
                  det_prob, det_noise_scale, 
                  death_prob, death_rate, det_prob_d)
        
        beta_f, x_f, y_f, z_f = SEIR_dynamics(T_future+1, params, x[-1,:], 
                                              suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        z = np.append(z, z_f)
        
    return beta, x, y, z, det_prob, death_prob
