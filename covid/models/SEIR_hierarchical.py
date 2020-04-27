import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from functools import partial
from ..compartment import SEIRModel
from ..glm import glm, GLM, log_link, logit_link, Gamma, Beta

from .util import observe, ExponentialRandomWalk, get_future_data

"""
************************************************************
SEIR hierarchical
************************************************************
"""


def SEIR_dynamics_hierarchical(T, params, x0, obs = None, use_rw = True, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
        
    beta, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift = params

    num_places, T_minus_1 = beta.shape
    assert(T_minus_1 == T-1)
    
    # prep for broadcasting over time
    sigma = sigma[:,None]
    gamma = gamma[:,None]
    det_rate = det_rate[:,None]
    
    if use_rw:
        with numpyro.plate("places", num_places):
            rw = numpyro.sample("rw" + suffix,
                                ExponentialRandomWalk(loc = rw_loc,
                                                      scale = rw_scale,
                                                      drift = drift, 
                                                      num_steps = T-1))
    else:
        rw = rw_loc

    beta *= rw

    # Run ODE
    apply_model = lambda x0, beta, sigma, gamma: SEIRModel.run(T, x0, (beta, sigma, gamma))
    x = jax.vmap(apply_model)(x0, beta, sigma, gamma)

    x = x[:,1:,:] # drop first time step from result (duplicates initial value)
    numpyro.deterministic("x" + suffix, x)
   
    # Noisy observations
    y = observe("y" + suffix, x[:,:,4], det_rate, det_noise_scale, obs = obs)

    return rw, x, y


def SEIR_hierarchical(data = None,
                      place_data = None,
                      T_future = 0,
                      E_duration_est = 4.5,
                      I_duration_est = 3.0,
                      R0_est = 4.5,
                      det_rate_est = 0.3,
                      det_rate_conc = 100,
                      use_rw = True,
                      rw_scale = 1e-1,
                      det_noise_scale = 0.2,
                      drift_scale = None,
                      use_obs = False):

    '''
    Stochastic SEIR model. Draws random parameters and runs dynamics.
    '''
    
    num_places, _ = place_data.shape
    
    '''Generate R0'''
    ## TODO: forecasting with splines not yet supported b/c patsy will not evaluate
    ## splines outside of the outermost knots. Look into workaround/fix for this
    R0_glm = GLM("1 + C(state, OneHot) + state_of_emergency + shelter_in_place + Q('non-contact_school') + standardize(popdensity) + state : cr(t, df=3)", 
                 data, 
                 log_link,
                 partial(Gamma, var=0.1),
                 prior = dist.Normal(0, 0.1),
                 guess=R0_est,
                 name="R0")
    
    R0 = R0_glm.sample(shape=(num_places,-1))[0]
    
    '''Generate E_duration'''
    E_duration = glm('1 + C(state, OneHot)', 
                     place_data,
                     log_link,
                     partial(Gamma, var=0.05),
                     prior = dist.Normal(0, 0.05),
                     guess=E_duration_est,
                     name="E_duration")[0]
    
    '''Generate I_duration'''
    I_duration = glm('1 + C(state, OneHot)', 
                     place_data,
                     log_link,
                     partial(Gamma, var=0.05),
                     prior=dist.Normal(0, 0.05),
                     guess=I_duration_est,
                     name="I_duration")[0]


    '''Generate det_rate'''
    det_rate = glm('1 + C(state, OneHot)',
                   place_data,
                   logit_link,
                   partial(Beta, conc=det_rate_conc),
                   prior=dist.Normal(0, 0.025),
                   guess=det_rate_est,
                   name="det_rate")[0]
    
    # Broadcast to correct size
    N = np.array(place_data['totalpop'])
    
    R0 = R0.reshape((num_places, -1))
    _, T = R0.shape
    
    det_rate = det_rate
    sigma = 1/E_duration
    gamma = 1/I_duration
    beta = R0 * gamma[:,None]
    
    #beta = beta[:,:-1] # truncate to T-1 timesteps (transitions)
    
    with numpyro.plate("num_places", num_places): 
        I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
        E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))


    if use_obs:
        pos = np.array(data['positive']).reshape(num_places, T)
        obs0, obs = pos[:,0], pos[:,1:]
    else:
        obs0, obs = None, None
    
    '''
    Run model for each place
    '''
    x0 = jax.vmap(SEIRModel.seed)(N, I0, E0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    
    # First observation
    y0 = observe("y0", x0[:,4], det_rate, det_noise_scale, obs=obs0)

    # Run dynamics
    drift = 0.
    rw_loc = 1.
    params = (beta[:,:-1], sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift)
    rw, x, y = SEIR_dynamics_hierarchical(T, params, x0, 
                                          use_rw = use_rw, 
                                          obs = obs)
    
    x = np.concatenate((x0[:,None,:], x), axis=1)
    y = np.concatenate((y0[:,None], y), axis=1)
    
    if T_future > 0:
        
        future_data = get_future_data(data, T_future-1)
        
        R0_future = R0_glm.sample(future_data, name="R0_future", shape=(num_places,-1))[0]

        beta_future = R0_future * gamma[:, None]
        beta_future = np.concatenate((beta[:,-1,None], beta_future), axis=1)
        
        rw_loc = rw[:,-1,None] if use_rw else 1.
        
        params = (beta_future, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift)
        
        _, x_f, y_f = SEIR_dynamics_hierarchical(T_future+1, 
                                                 params, 
                                                 x[:,-1,:], 
                                                 use_rw = use_rw,
                                                 suffix="_future")
        
        x = np.concatenate((x, x_f), axis=1)
        y = np.concatenate((y, y_f), axis=1)
        
    return beta, x, y, det_rate