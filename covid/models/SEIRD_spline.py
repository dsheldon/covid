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

def SEIR_dynamics(T, T_future, params, x0, obs=None, death=None, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
    
    beta, sigma, gamma, rw_scale, drift, \
    det_rate, det_noise_scale, hosp_rate, death_rate, det_rate_d , beta0_glm = params

    #beta = numpyro.sample("beta" + suffix,
     #             ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))
    
    
    def exp_t(t):
        return .25**t


    if suffix != "_future":
        data = pd.DataFrame({'t':np.arange(T-1)})
        beta = beta0_glm.sample(data, name="b0" + suffix, shape=(-1))[0]
    else:
        data = pd.DataFrame({'t':np.arange(T_future-1)})
        beta = beta0_glm.sample(data, name="b0" + suffix, shape=(-1))[0]
        beta = beta*exp_t(np.arange(T_future-1))
   
    # Run ODE
    if suffix != "_future":
        x = SEIRModel.run(T, x0, (beta, sigma, gamma, hosp_rate, death_rate))
    else:
        x = SEIRModel.run(T_future, x0, (beta, sigma, gamma, hosp_rate, death_rate))

    x = x[1:] # first entry duplicates x0
    numpyro.deterministic("x" + suffix, x)

    latent = x[:,6] # cumulative cases

    # Noisy observations
    y = observe("y" + suffix, x[:,6], det_rate, det_noise_scale, obs = obs)
   
    z = observe("z" + suffix, x[:,5], det_rate_d, det_noise_scale/4., obs = death)
  
        
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
                    death=None,
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
    H0 = numpyro.sample("H0", dist.Uniform(0, 0.02*N))
    D0 = numpyro.sample("D0", dist.Uniform(0, 0.02*N))

    # Sample parameters
    sigma = numpyro.sample("sigma", 
                           dist.Gamma(sigma_shape, sigma_shape * E_duration_est))
    
    gamma = numpyro.sample("gamma", 
                           dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

    data = pd.DataFrame({'t':np.arange(T-1)})

    beta0_glm = GLM("1 + cr(t, df=5) ", 
                 data, 
                 log_link,
                 partial(Gamma, var=0.1),
                 prior = dist.Normal(0, 0.1),
                 guess=.95,
                 name="beta0")
    
    det_rate = numpyro.sample("det_rate", 
                              dist.Beta(det_rate_est * det_rate_conc,
                                        (1-det_rate_est) * det_rate_conc))
    det_rate_d = numpyro.sample("det_rate_d", 
                               dist.Beta(.9 * 10,
                                        (1-.9) * 10))
    
    hosp_rate = numpyro.sample("hosp_rate", 
                              dist.Beta(.1 * 10,
                                        (1-.1) * 10))
    death_rate = numpyro.sample("death_rate", 
                             dist.Beta(.1 * 10
,
                                        (1-.1) * 10))
    

  
    
    
    if False:#drift_scale is not None:
        drift = numpyro.sample("drift", dist.Normal(loc=-.5, scale=1e-5))
    else:
        drift = 0
        
    
    x0 = SEIRModel.seed(N, I0, E0,H0,D0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    death0, death = (None, None) if death is None else (death[0], death[1:])

    
    # First observation
    y0 = observe("y0", x0[6], det_rate, det_noise_scale, obs=obs0)
    z0 = observe("z0", x0[5], det_rate_d, det_noise_scale, obs=death0)

    beta = .95
        
    params = (beta, sigma, gamma, 
              rw_scale, drift, 
              det_rate, det_noise_scale, 
              hosp_rate,death_rate,det_rate_d,beta0_glm)
    
    beta, x, y, z = SEIR_dynamics(T, 0, params, x0, 
                                  obs = obs, 
                                  death = death)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    z = np.append(z0, z)
    
    if T_future > 0:
        
        params = (beta[-1], sigma, gamma, 
                  rw_scale, drift, 
                  det_rate, det_noise_scale, 
                  hosp_rate, death_rate, det_rate_d,beta0_glm)
        
        beta_f, x_f, y_f, z_f = SEIR_dynamics(T,T_future+1, params, x[-1,:], 
                                              suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        z = np.append(z, z_f)
        
    return beta, x, y, z, det_rate, hosp_rate

