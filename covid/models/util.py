import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

import pandas as pd

import warnings


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
    
    log_loc = np.log(loc) + drift * (np.arange(num_steps)+0.)
    
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = log_loc, scale=1.),
            dist.transforms.ExpTransform()
        ]
    )

def LogisticRandomWalk(loc=1., scale=1e-2, drift=0., num_steps=100):
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
   
    logistic_loc = np.log(loc/(1-loc)) + drift * (np.arange(num_steps)+0.)
   
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = logistic_loc, scale=1.),
            dist.transforms.SigmoidTransform()
        ]
    )


def NB2(mu=None, k=None):
    conc = 1./k
    rate = conc/mu
    return dist.GammaPoisson(conc, rate)


def observe(*args, **kwargs):
    return observe_normal(*args, **kwargs)
#    return observe_poisson(*args, **kwargs)
#    return observe_gamma(*args, **kwargs)

def observe_normal(name, latent, det_rate, det_noise_scale, obs=None):
    mask = True

    reg = 0.
    latent = latent + (reg/det_rate)
    
    if obs is not None:
        mask = np.isfinite(obs) & (obs >= 0)
        obs = np.where(mask, obs, 0.0)
        obs += reg
        
    det_rate = np.broadcast_to(det_rate, latent.shape)

    mean = det_rate * latent
    scale = det_noise_scale * mean + 1
    d = dist.TruncatedNormal(0., mean, scale)
    
    numpyro.deterministic("mean_" + name, mean)
    
    with numpyro.handlers.mask(mask_array=mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y


def observe_poisson(name, latent, det_prob, obs=None):

    mask = True
    if obs is not None:
        mask = np.isfinite(obs) & (obs >= 0)
        obs = np.where(mask, obs, 0.0)
        
    det_prob = np.broadcast_to(det_prob, latent.shape)

    mean = det_prob * latent
    d = dist.Poisson(mean)    
    numpyro.deterministic("mean_" + name, mean)
    
    with numpyro.handlers.mask(mask_array=mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y


def observe_nb2(name, latent, det_prob, dispersion, obs=None):

    mask = True
    if obs is not None:
        mask = np.isfinite(obs) & (obs >= 0.0)
        obs = np.where(mask, obs, 0.0)
        
    if np.any(mask):
        warnings.warn('Some observed values are invalid')
        
    det_prob = np.broadcast_to(det_prob, latent.shape)

    mean = det_prob * latent
    numpyro.deterministic("mean_" + name, mean)
    
    d = NB2(mu=mean, k=dispersion)
    
    with numpyro.handlers.mask(mask_array=mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y


"""
************************************************************
Data handling within model
************************************************************
"""

def get_future_data(data, T, offset=1):
    '''Projects data frame with (place, time) MultiIndex into future by
       repeating final time value for each place'''
    data = data.unstack(0)
    orig_start = data.index.min()
    start = data.index.max() + pd.Timedelta(offset, "D")
    future = pd.date_range(start=start, periods=T, freq="D")
    data = data.reindex(future, method='nearest')
    data['t'] = (data.index-orig_start)/pd.Timedelta("1d")
    data = data.stack()
    data.index = data.index.swaplevel(0, 1)
    data = data.sort_index()
    return data
    


