import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from functools import partial
from ..compartment import SEIRModel
from ..glm import glm, GLM, log_link, logit_link, Gamma, Beta

from .util import observe, ExponentialRandomWalk, future_data


"""
************************************************************
SEIRD model
************************************************************
"""

def SEIRD_dynamics(T, params, x0, obs=None, death=None, suffix=""):
    '''Run SEIRD dynamics for T time steps'''
    
    beta0, sigma, gamma, rw_scale, drift, \
    det_prob, det_noise_scale, death_prob, death_rate, det_prob_d  = params

    beta = numpyro.sample("beta" + suffix,
                  ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

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


def SEIRD_stochastic(T = 50,
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
        drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
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



"""
************************************************************
SEIR hierarchical
************************************************************
"""


def SEIR_dynamics_hierarchical(T, params, x0, obs = None, death=None, use_rw = True, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
        
    beta, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift, hosp_rate, death_rate, det_rate_d = params

    num_places, T_minus_1 = beta.shape
    assert(T_minus_1 == T-1)
    
    # prep for broadcasting over time
    sigma = sigma[:,None]
    gamma = gamma[:,None]
    det_rate = det_rate[:,None]
    det_rate_d = det_rate_d[:,None]

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
    apply_model = lambda x0, beta, sigma, gamma, hosp_rate, death_rate: SEIRModel.run(T, x0, (beta, sigma, gamma, hosp_rate, death_rate))
    x = jax.vmap(apply_model)(x0, beta, sigma, gamma, hosp_rate, death_rate)

    x = x[:,1:,:] # drop first time step from result (duplicates initial value)
    numpyro.deterministic("x" + suffix, x)
   
    # Noisy observations
    y = observe("y" + suffix, x[:,:,6], det_rate, det_noise_scale, obs = obs)
    z = observe("z" + suffix, x[:,:,5], det_rate_d, det_noise_scale, obs = death)

    return rw, x, y, z


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
    
    det_rate_d = glm('1 + C(state, OneHot)',
                   place_data,
                   logit_link,
                   partial(Beta, conc=det_rate_conc),
                   prior=dist.Normal(0, 0.1),
                   guess=.95,
                   name="det_rate_d")[0]
    
    
    death_rate = glm('1 + C(state, OneHot)',
                     place_data,
                     log_link,
                     partial(Gamma, var=0.05),
                     prior=dist.Normal(0, 0.5),
                     guess=I_duration_est,
                     name="death_rate")[0]



    hosp_rate = glm('1 + C(state, OneHot)',
                     place_data,
                     log_link,
                     partial(Beta, conc=100),
                     prior=dist.Normal(0, 0.5),
                     guess=I_duration_est,
                     name="hosp_rate")[0]
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
        death_obs = np.array(data['death']).reshape(num_places, T)
        death0, death = death_obs[:,0], death_obs[:,1:]
    else:
        obs0, obs = None, None
        death0, death = None, None
    '''
    Run model for each place
    '''
    x0 = jax.vmap(SEIRModel.seed)(N, I0, E0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    
    # First observation
    y0 = observe("y0", x0[:,6], det_rate, det_noise_scale, obs=obs0)
    z0 = observe("z0", x0[:,5], det_rate_d, det_noise_scale, obs=death0)

    # Run dynamics
    drift = 0.
    rw_loc = 1.
    params = (beta[:,:-1], sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift, hosp_rate,death_rate, det_rate_d)
    rw, x, y, z = SEIR_dynamics_hierarchical(T, params, x0, 
                                          use_rw = use_rw, 
                                          obs = obs,
                                            death=death)
    
    x = np.concatenate((x0[:,None,:], x), axis=1)
    y = np.concatenate((y0[:,None], y), axis=1)
    z = np.concatenate((z0[:,None], z), axis=1)

    if T_future > 0:
        
        future_data = future_data(data, T_future-1)
        
        R0_future = R0_glm.sample(future_data, name="R0_future", shape=(num_places,-1))[0]

        beta_future = R0_future * gamma[:, None]
        beta_future = np.concatenate((beta[:,-1,None], beta_future), axis=1)
        
        #rw_loc = rw[:,-1,None] if use_rw else 1.
        
        params = (beta_future, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift, hosp_rate, death_rate, det_rate_d)
        
        _, x_f, y_f, z_f = SEIR_dynamics_hierarchical(T_future+1, 
                                                 params, 
                                                 x[:,-1,:], 
                                                 use_rw = use_rw,
                                                 suffix="_future")
        
        x = np.concatenate((x, x_f), axis=1)
        y = np.concatenate((y, y_f), axis=1)
        z = np.concatenate((z, z_f), axis=1)

        
    return beta, x, y,z, det_rate
