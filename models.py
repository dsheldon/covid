import numpy as onp

import jax
import jax.numpy as np
from jax.experimental.ode import odeint

import numpyro
import numpyro.distributions as dist

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from compartment import SIRModel, SEIRModel

import util


"""
************************************************************
Shared model / distribution components
************************************************************
"""

def BinomialApprox(n, p, conc=None):
    '''
    Return distribution that is a continuous approximation to 
    Binomial(n, p); allows overdispersion by setting conc < n
    '''
    if conc is None:
        conc = n
        
    a = conc * p
    b = conc * (1-p)
    
    # This is the distribution of n * Beta(a, b)
    return dist.TransformedDistribution(
        dist.Beta(a, b),
        dist.transforms.AffineTransform(loc = 0, scale = n)
    )


def ExponentialRandomWalk(loc=1., scale=1e-2, drift=0., num_steps=100):
    '''
    Return distrubtion of exponentiated Gaussian random walk
    
    Variables are x_0, ..., x_{T-1}
    
    Dynamics in log-space are random walk with drift:
       log(x_0) := log(loc) 
       log(x_t) := log(x_{t-1}) + drift + eps_t,    eps_t ~ N(0, scale)
        
    ==> Dynamics in non-log space are:
        x_0 := loc
        x_t := x_{t-1} * exp(drift + eps_t),    eps_t ~ N(0, scale)        
    '''
    
    log_loc = np.log(loc) + drift * np.arange(num_steps, dtype='float32')
    
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = log_loc, scale=1.),
            dist.transforms.ExpTransform()
        ]
    )



def observe(*args, **kwargs):
#    return _observe_binom_approx(*args, **kwargs)
    return _observe_normal(*args, **kwargs)

def _observe_normal(name, latent, det_rate, det_noise_scale, obs=None):
    mask = True

    reg = 0.5
    latent = latent + (reg/det_rate)
    
    if obs is not None:
        mask = np.isfinite(obs)
        obs = np.where(mask, obs, 0.0)
        obs += reg
        
    det_rate = np.broadcast_to(det_rate, latent.shape)

    mean = det_rate * latent
    scale = det_noise_scale * mean + 1
    d = dist.Normal(mean, scale)
    
    numpyro.deterministic("mean_" + name, mean)
    
    with numpyro.handlers.mask(mask_array=mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y

    
def _observe_binom_approx(name, latent, det_rate, det_conc, obs=None):
    '''Make observations of a latent variable using BinomialApprox.'''
    
    mask = True
    
    # Regularization: add reg to observed, and (reg/det_rate) to latent
    # The primary purpose is to avoid zeros, which are invalid values for 
    # the Beta observation model.
    reg = 0.5 
    latent = latent + (reg/det_rate)
        
    if obs is not None:
        '''
        Workaround for a jax issue: substitute default values
        AND mask out bad observations. 
        
        See https://forum.pyro.ai/t/behavior-of-mask-handler-with-invalid-observation-possible-bug/1719/5
        '''
        mask = np.isfinite(obs)
        obs = np.where(mask, obs, 0.5 * latent)
        obs = obs + reg

    det_rate = np.broadcast_to(det_rate, latent.shape)        
    det_conc = np.minimum(det_conc, latent) # don't allow it to be *more* concentrated than Binomial
    
    d = BinomialApprox(latent + (reg/det_rate), det_rate, det_conc)
    
    with numpyro.handlers.mask(mask_array=mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y



"""
************************************************************
SEIR model
************************************************************
"""

def SEIR_dynamics(T, params, x0, obs=None, death=None, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
    
    beta0, sigma, gamma, rw_scale, drift, \
    det_rate, det_noise_scale, hosp_rate, death_rate, det_rate_d  = params

    beta = numpyro.sample("beta" + suffix,
                  ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

    # Run ODE
    x = SEIRModel.run(T, x0, (beta, sigma, gamma, hosp_rate, death_rate))
    x = x[1:] # first entry duplicates x0
    numpyro.deterministic("x" + suffix, x)

    latent = x[:,6] # cumulative cases

    # Noisy observations
    y = observe("y" + suffix, x[:,6], det_rate, det_noise_scale, obs = obs)
   
    z = observe("z" + suffix, x[:,5], det_rate_d, det_noise_scale, obs = death)
  
        
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
    det_rate_d = numpyro.sample("det_rate_d", 
                               dist.Beta(.9 * 100,
                                        (1-.9) * 100))
    
    hosp_rate = numpyro.sample("hosp_rate", 
                              dist.Beta(.1 * 100,
                                        (1-.1) * 100))
    death_rate = numpyro.sample("death_rate", 
                             dist.Beta(.1 * 100,
                                        (1-.1) * 100))
    

  
    
    
    if drift_scale is not None:
        drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
    else:
        drift = 0
        
    
    x0 = SEIRModel.seed(N, I0, E0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    death0, death = (None, None) if death is None else (death[0], death[1:])

    
    # First observation
    y0 = observe("y0", x0[6], det_rate, det_noise_scale, obs=obs0)
    z0 = observe("z0", x0[5], det_rate_d, det_noise_scale, obs=death0)

   
        
    params = (beta0, sigma, gamma, 
              rw_scale, drift, 
              det_rate, det_noise_scale, 
              hosp_rate,death_rate,det_rate_d)
    
    beta, x, y, z = SEIR_dynamics(T, params, x0, 
                                  obs = obs, 
                                  death = death)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    z = np.append(z0, z)
    
    if T_future > 0:
        
        params = (beta[-1], sigma, gamma, 
                  rw_scale, drift, 
                  det_rate, det_noise_scale, 
                  hosp_rate, death_rate, det_rate_d)
        
        beta_f, x_f, y_f, z_f = SEIR_dynamics(T_future+1, params, x[-1,:], 
                                              suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        z = np.append(z, z_f)
        
    return beta, x, y, z, det_rate, hosp_rate



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


from functools import partial
from glm import glm, GLM, log_link, logit_link, Gamma, Beta

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
        
        future_data = util.future_data(data, T_future-1)
        
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



"""
************************************************************
Plotting
************************************************************
"""

def combine_samples(samples, fields=['x', 'y', 'z', 'mean_y', 'mean_z']):
    '''Combine x0, x, x_future and similar fields into a single array'''

    samples = samples.copy()
    
    for f in fields:
        if f in samples:
            f0, f_future = f + '0', f + '_future'
            data = np.concatenate((samples[f0][:,None], samples[f]), axis=1)
            del samples[f0]
            
            if f_future in samples:
                data = np.concatenate((data, samples[f_future]), axis=1)
                del samples[f_future]
            
            samples[f] = data
    
    return samples


def plot_samples(samples, 
                 plot_fields=['I', 'y'], 
                 T=None, 
                 t=None, 
                 ax=None, 
                 n_samples=0,
                 legend=True,
                 model='SEIR'):
    '''
    Plotting method for SIR-type models. 
    (Needs some refactoring to handle both SIR and SEIR)
    '''

    samples = combine_samples(samples)
    n_samples = np.minimum(n_samples, samples['x'].shape[0])
    
    T_data = samples['x'].shape[1]
    if T is None or T > T_data:
        T = T_data
    
    if model == 'SIR':
        S, I, R, C = 0, 1, 2, 3
    elif model == 'SEIR':
        S, E, I, R, H, D, C = 0, 1, 2, 3, 4, 5, 6
    else:
        raise ValueError("Bad model")
    
    fields = {'susceptible'     : samples['x'][:,:T, S],
              'infectious'      : samples['x'][:,:T, I],
              'removed'         : samples['x'][:,:T, R],
              'total infectious': samples['x'][:,:T, C],
              'total confirmed' : samples['y'][:,:T],
              'total deaths'    : samples['z'][:,:T],
              'daily confirmed' : onp.diff(samples['mean_y'][:,:T], axis=1, prepend=np.nan),
              'daily deaths'    : onp.diff(samples['mean_z'][:,:T], axis=1, prepend=np.nan)}
                  
    fields = {k: fields[k] for k in plot_fields}

    medians = {k: np.median(v, axis=0) for k, v in fields.items()}    
    pred_intervals = {k: np.percentile(v, (10, 90), axis=0) for k, v in fields.items()}
    
    t = np.arange(T) if t is None else t[:T]
    
    ax = ax if ax is not None else plt.gca()
    
    df = pd.DataFrame(index=t, data=medians)
    df.plot(ax=ax, legend=legend)
    median_max = df.max().values
    
    colors = [l.get_color() for l in ax.get_lines()]
    
    # Add individual field lines
    if n_samples > 0:
        i = 0
        for k, data in fields.items():
            step = np.array(data.shape[0]/n_samples).astype('int32')
            df = pd.DataFrame(index=t, data=data[::step,:].T)
            df.plot(ax=ax, lw=0.25, color=colors[i], alpha=0.25, legend=False)
            i += 1
    
    # Add prediction intervals
    pi_max = 10
    i = 0
    for k, pred_interval in pred_intervals.items():
        ax.fill_between(t, pred_interval[0,:], pred_interval[1,:], color=colors[i], alpha=0.1, label='CI')
        pi_max = np.maximum(pi_max, np.nanmax(pred_interval[1,:]))
        i+= 1
    
    return median_max, pi_max
    
    
def plot_forecast(post_pred_samples, T, confirmed, 
                  t = None, 
                  scale='log',
                  n_samples= 100,
                  death = None,
                  daily = False,
                  **kwargs):

    t = t if t is not None else np.arange(T)

    fig, axes = plt.subplots(nrows = 2, figsize=(8,12), sharex=True)
    
    
    if daily:
        variables = ['daily confirmed', 'daily deaths']
        w = 7
        min_periods = 1
        observations = [confirmed.diff().rolling(w, min_periods=min_periods, center=True).mean(), 
                        death.diff().rolling(w, min_periods=min_periods, center=True).mean()]
    else:
        variables = ['total confirmed', 'total deaths']
        observations = [confirmed, death]
        
    for variable, observation, ax in zip(variables, observations, axes):
    
        median_max, pi_max = plot_samples(post_pred_samples, T=T, t=t, ax=ax, plot_fields=[variable], **kwargs)
        observation.plot(ax=ax, style='o')
    
        ax.axvline(observation.index.max(), linestyle='--', alpha=0.5)
        ax.grid(axis='y')

        if scale == 'log':
            ax.set_yscale('log')

            # Don't display below 1
            bottom, top = ax.get_ylim()
            bottom = 1 if bottom < 1 else bottom
            ax.set_ylim([bottom, pi_max])
        else:
            top = np.minimum(2*median_max, pi_max)
            ax.set_ylim([0, top])

    return fig, ax

def plot_R0(mcmc_samples, start):

    fig = plt.figure(figsize=(5,3))
    
    # Compute average R0 over time
    gamma = mcmc_samples['gamma'][:,None]
    beta = mcmc_samples['beta']
    t = pd.date_range(start=start, periods=beta.shape[1], freq='D')
    R0 = beta/gamma

    pi = np.percentile(R0, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'R0': np.median(R0, axis=0)})
    df.plot(style='-o')
    plt.fill_between(t, pi[0,:], pi[1,:], alpha=0.1)

    plt.axhline(1, linestyle='--')
    
    #plt.tight_layout()

    return fig
