import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

import pandas as pd
import matplotlib.pyplot as plt

from compartment import SIRModel, SEIRModel


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


def ExponentialRandomWalk(loc=1., scale=1e-2, num_steps=100):
    '''
    Return distrubtion of exponentiated Gaussian random walk
    '''
    
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = np.log(loc), scale=1.),
            dist.transforms.ExpTransform()
        ]
    )



def observe(name, latent, det_rate, det_conc, obs=None):
    '''
    Make observations of a latent variable using the BinomialApprox
    noise model. Wrapper that handles broadcasting and replacement
    of bad values.
    '''
    mask = True
    
    if obs is not None:
        '''
        Workaround for a jax issue: substitute default values
        AND mask out bad observations. 
        
        See https://forum.pyro.ai/t/behavior-of-mask-handler-with-invalid-observation-possible-bug/1719/5
        '''
        mask = (obs > 0) & (obs < latent)  
        obs = np.where(mask, obs, 0.5 * latent)
         
    # This ensure there is a separate RV for each latent variable
    det_rate = np.broadcast_to(det_rate, latent.shape)
    
    d = BinomialApprox(latent, det_rate, det_conc)

    with numpyro.handlers.mask(mask):
        y = numpyro.sample(name, d, obs = obs)
        
    return y



"""
************************************************************
SIR model
************************************************************
"""

def SIR_dynamics(SIR, T, params, x0, obs = None, suffix=""):

    '''
    Run SIR dynamics for T time steps
    '''
    
    beta0, gamma, det_rate, det_conc, drift_scale = params

    beta = numpyro.sample("beta" + suffix, 
                  ExponentialRandomWalk(loc = beta0, scale=drift_scale, num_steps=T-1))

    # Run ODE
    x = SIR.run(T, x0, (beta, gamma))
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
    

    SIR = SIRModel()
    x0 = SIR.seed(N, I0)
    numpyro.deterministic("x0", x0)

    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    
    # First observation
    y0 = observe("y0", x0[3], det_rate, det_conc, obs=obs0)
    
    # Run dynamics
    params = (beta0, gamma, det_rate, det_conc, drift_scale)
    
    beta, x, y = SIR_dynamics(SIR, T, params, x0, obs = obs)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    
    if T_future > 0:
        
        params = (beta[-1], gamma, det_rate, det_conc, drift_scale)
        
        beta_f, x_f, y_f = SIR_dynamics(SIR, T_future+1, params, x[-1,:], suffix="_future")
        
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
    
    beta0, gamma, det_rate, det_conc, drift_scale = params

    # Add a dimension to these for broadcasting with 2d arrays (num_places x T)
    beta0 = beta0[:,None]
    det_rate = det_rate[:,None]
    
    with numpyro.plate("num_places", beta0.shape[0]):
        beta = numpyro.sample("beta" + suffix, 
                      ExponentialRandomWalk(loc = beta0, scale=drift_scale, num_steps=T-1))
    
    
    # Run ODE
    apply_model = lambda x0, beta, gamma: SIR.run(T, x0, (beta, gamma))        

    # TODO: workaround for vmap bug
    #x = jax.vmap(apply_model)(x0, beta, gamma)
    x = np.stack([apply_model(xx, b, g) for xx, b, g in zip(x0, beta, gamma)])
                  
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
                     drift_scale = 5e-2,
                     obs = None):
    '''
    Hierarchical SIR model
    '''


    '''
    Draw shared parameters
    '''
    
#     gamma_ = numpyro.sample("gamma_", 
#                      dist.Gamma(gamma_shape, 
#                                 gamma_shape * duration_mean))

#     beta_ = numpyro.sample("beta_", 
#                              dist.Gamma(beta_shape, 
#                                  beta_shape * duration_mean/R0_mean))


#     det_rate_ = numpyro.sample("det_rate_", 
#                                dist.Beta(det_rate_mean * det_rate_conc,
#                                          (1 - det_rate_mean) * det_rate_conc))
    
        
    # Broadcast shared parameters to correct size
    N = np.broadcast_to(N, (num_places,))
    #gamma = np.broadcast_to(gamma, (num_places,))
    #print("N", N)
    

    '''
    Draw place-specific parameters
    '''
    with numpyro.plate("num_places", num_places):
        
        I0 = numpyro.sample("I0", dist.Uniform(0, N*0.02))
        
        gamma = numpyro.sample("gamma", 
                     dist.Gamma(gamma_shape, 
                                gamma_shape * duration_mean))

        beta0 = numpyro.sample("beta0", 
                             dist.Gamma(beta_shape, 
                                 beta_shape * duration_mean/R0_mean))

        det_rate = numpyro.sample("det_rate", 
                               dist.Beta(det_rate_mean * det_rate_conc,
                                         (1 - det_rate_mean) * det_rate_conc))

        
#         gamma = numpyro.sample("gamma", dist.Gamma(20, 20 / gamma_))
        
#         beta0 = numpyro.sample("beta0", dist.Gamma(20, 20 / beta_))
        
#         det_rate = numpyro.sample("det_rate", dist.Beta(100*det_rate_, 100*(1-det_rate_)))

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
    params = (beta0, gamma, det_rate, det_conc, drift_scale)
    beta, x, y = SIR_dynamics_hierarchical(SIR, T, params, x0, obs = obs)
    
    x = np.concatenate((x0[:,None,:], x), axis=1)
    y = np.concatenate((y0[:,None], y), axis=1)
    
    if T_future > 0:
        
        params = (beta[:,-1], gamma, det_rate, det_conc, drift_scale)
        
        beta_f, x_f, y_f = SIR_dynamics_hierarchical(SIR, 
                                                     T_future+1, 
                                                     params, x
                                                     [:,-1,:], 
                                                     suffix="_future")
        
        x = np.concatenate((x, x_f), axis=1)
        y = np.concatenate((y, y_f), axis=1)
        
    return beta, x, y, det_rate



"""
************************************************************
SEIR model
************************************************************
"""

def SEIR_dynamics(SEIR, T, params, x0, obs = None, suffix=""):

    '''
    Run SEIR dynamics for T time steps
    '''
    
    beta0, sigma, gamma, det_rate, det_conc, drift_scale = params

    beta = numpyro.sample("beta" + suffix, 
                  ExponentialRandomWalk(loc=beta0, scale=drift_scale, num_steps=T-1))

    # Run ODE
    x = SEIR.run(T, x0, (beta, sigma, gamma))
    x = x[1:] # first entry duplicates x0
    numpyro.deterministic("x" + suffix, x)

    latent = x[:,4]

    # Noisy observations
    y = observe("y" + suffix, x[:,4], det_rate, det_conc, obs = obs)

    return beta, x, y


def SEIR_stochastic(T = 50,
                    N = 1e5,
                    T_future = 0,
                    E_duration_mean = 4.0,
                    I_duration_mean = 1.5,
                    R0_mean = 3.5,
                    beta_shape = 1,
                    sigma_shape = 5,
                    gamma_shape = 5,
                    det_rate_mean = 0.3,
                    det_rate_conc = 50,
                    det_conc = 100,
                    drift_scale = 1e-1,
                    obs = None):

    '''
    Stochastic SEIR model. Draws random parameters and runs dynamics.
    '''

    # Sample initial number of infected individuals
    I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
    E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
        

    # Sample parameters
    sigma = numpyro.sample("sigma", 
                           dist.Gamma(sigma_shape, sigma_shape * E_duration_mean))


    gamma = numpyro.sample("gamma", 
                           dist.Gamma(gamma_shape, gamma_shape * I_duration_mean))    

    beta0 = numpyro.sample("beta0", 
                          dist.Gamma(beta_shape, beta_shape * I_duration_mean/R0_mean))
        
    det_rate = numpyro.sample("det_rate", 
                              dist.Beta(det_rate_mean * det_rate_conc,
                                        (1-det_rate_mean) * det_rate_conc))

    
    SEIR = SEIRModel()
    x0 = SEIR.seed(N, I0, E0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    obs0, obs = (None, None) if obs is None else (obs[0], obs[1:])
    
    # First observation
    y0 = observe("y0", x0[4], det_rate, det_conc, obs=obs0)
    
    params = (beta0, sigma, gamma, det_rate, det_conc, drift_scale)
    
    beta, x, y = SEIR_dynamics(SEIR, T, params, x0, obs = obs)
    
    x = np.vstack((x0, x))
    y = np.append(y0, y)
    
    if T_future > 0:
        
        params = (beta[-1], sigma, gamma, det_rate, det_conc, drift_scale)
        
        beta_f, x_f, y_f = SEIR_dynamics(SEIR, T_future+1, params, x[-1,:], suffix="_future")
        
        x = np.vstack((x, x_f))
        y = np.append(y, y_f)
        
    return beta, x, y, det_rate


"""
************************************************************
Plotting
************************************************************
"""

def plot_samples(samples, plot_fields=['I', 'y'], T=None, t=None, ax=None, model='SIR'):
    '''
    Plotting method for SIR-type models. 
    (Needs some refactoring to handle both SIR and SEIR)
    '''
    
    T_data = samples['x'].shape[1] + 1
    if 'x_future' in samples:
        T_data += samples['x_future'].shape[1]
    
    if T is None or T > T_data:
        T = T_data

    x0 = samples['x0'][:, None]
    x = samples['x']
    x = np.concatenate((x0, x), axis=1)

    if 'x_future' in samples:
        x_future = samples['x_future']
        x = np.concatenate((x, x_future), axis=1)
    
    labels = {
        'S': 'susceptible',
        'I': 'infected',
        'R': 'removed',
        'C': 'confirmed'
    }

    if model == 'SIR':
        S, I, R, C = 0, 1, 2, 3
    elif model == 'SEIR':
        S, E, I, R, C = 0, 1, 2, 3, 4
    else:
        raise ValueError("Bad model")
    
    fields = {'S': x[:,:T, S],
              'I': x[:,:T, I],
              'R': x[:,:T, R],
              'C': x[:,:T, C]}    
    
    if 'y' in samples:
        y0 = samples['y0'][:, None]
        y = samples['y']
        y = np.concatenate((y0, y), axis=1)
        if 'y_future' in samples:
            y_future = samples['y_future']
            y = np.concatenate((y, y_future), axis=1)
        fields['y'] = y[:,:T]
    
    fields = {k: fields[k] for k in plot_fields}

    means = {k: v.mean(axis=0) for k, v in fields.items()}
    
    pred_intervals = {k: np.percentile(v, (10, 90), axis=0) for k, v in fields.items()}
    
    # Use pandas to plot means (for better date handling)
    if t is None:
        t = np.arange(T)
    else:
        t = t[:T]

    df = pd.DataFrame(index=t, data=means)
    df.plot(ax=ax)
    
    # Add prediction intervals
    for k, pred_interval in pred_intervals.items():
        ax = ax if ax is not None else plt.gca()
        ax.fill_between(t, pred_interval[0,:], pred_interval[1,:], alpha=0.1)
        
    plt.ylim([pred_intervals['I'][0,:].min(), pred_intervals['I'][1,:].max()])