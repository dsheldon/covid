import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

import pandas as pd

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
   
    logistic_loc = np.log(loc/(1-loc)) + drift * np.arange(num_steps, dtype='float32')
   
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = logistic_loc, scale=1.),
            dist.transforms.SigmoidTransform()
        ]
    )



def observe(*args, **kwargs):
#    return _observe_binom_approx(*args, **kwargs)
    return _observe_normal(*args, **kwargs)

def _observe_normal(name, latent, det_rate, det_noise_scale, obs=None):
    mask = True

    reg = 0.
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
    


