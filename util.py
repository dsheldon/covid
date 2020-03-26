import jax.numpy as np
import numpyro.distributions as dist
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_samples(samples, plot_fields=['I', 'y'], T=None, t=None, ax=None):
    
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
    
    fields = {'S': x[:,:T,0],
              'I': x[:,:T,1],
              'R': x[:,:T,2],
              'C': x[:,:T,3]}    
    
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