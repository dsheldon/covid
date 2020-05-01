import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt

from jax.random import PRNGKey


"""
************************************************************
Base class for models
************************************************************
"""

class Model():
    
    names = {
        'S': 'susceptible',
        'I': 'infectious',
        'R': 'removed',
        'E': 'exposed',
        'H': 'hospitalized',
        'D': 'dead',
        'C': 'cumulative infected',
        'y': 'confirmed',
        'z': 'deaths'
    }
            
    
    def __init__(self, mcmc_samples=None, **args):
        self.mcmc_samples = mcmc_samples
        self.obs_args = {} # set by subclass
        self.args = args
        
    
    """
    ***************************************
    Inference and sampling routines
    ***************************************
    """
    
    def infer(self, num_warmup=1000, num_samples=1000, num_chains=1, rng_key=PRNGKey(1), **args):
        '''Fit using MCMC'''
        
        args = dict(self.args, **args, **self.obs_args)
        
        kernel = NUTS(self, init_strategy = numpyro.infer.util.init_to_median())

        mcmc = MCMC(kernel, 
                    num_warmup=num_warmup, 
                    num_samples=num_samples, 
                    num_chains=num_chains)
     
        print(" * running MCMC")
        
        mcmc.run(rng_key, **args)    
        mcmc.print_summary()
        
        self.mcmc = mcmc
        self.mcmc_samples = mcmc.get_samples()
    
        return self.mcmc_samples
    
    
    def prior(self, num_samples=1000, rng_key=PRNGKey(2), **args):
        
        predictive = Predictive(self, posterior_samples={}, num_samples=num_samples)        
        
        args = dict(self.args, **args) # passed args take precedence        
        self.prior_samples = predictive(rng_key, **args)
        
        return self.prior_samples
    
    
    def predictive(self, rng_key=PRNGKey(3), **args):

        if self.mcmc_samples is None:
            raise RuntimeError("run inference first")

        predictive = Predictive(self, posterior_samples=self.mcmc_samples)

        args = dict(self.args, **args)
        return predictive(rng_key, **args)
    
    
    def forecast(self, num_samples=1000, rng_key=PRNGKey(4), **args):
        if self.mcmc_samples is None:
            raise RuntimeError("run inference first")

        predictive = Predictive(self, posterior_samples=self.mcmc_samples)

        args = dict(self.args, **args, **self.obs_args)        
        return predictive(rng_key, **args)
        
    
    """
    ***************************************
    Data access and plotting
    ***************************************
    """    
    
    @classmethod
    def combine_samples(cls, samples, f):
        '''Combine fields like x0, x, x_future into a single array'''
        
        f0, f_future = f + '0', f + '_future'
        data = np.concatenate((samples[f0][:,None], samples[f]), axis=1)
        if f_future in samples:
            data = np.concatenate((data, samples[f_future]), axis=1)
        return data
    
    
    @classmethod
    def get(cls, samples, c, **kwargs):
        
        daily = kwargs.get('daily', False)
        forecast = kwargs.get('forecast', False)
        
        if c in cls.compartments:
            x = samples['x_future'] if forecast else cls.combine_samples(samples, 'x')
            if daily:
                x = onp.diff(x, prepend=0.)
                
            j = cls.compartments.index(c)
            return x[:,:,j]
        
        else:
            return getattr(cls, c)(samples, **kwargs)  # call method named c
        
        
    @classmethod
    def z(cls, samples, forecast=False, daily=False):
        return samples['z_future'] if forecast else cls.combine_samples(samples, 'z')
    

    @classmethod
    def y(cls, samples, forecast=False, daily=False):      
        return samples['y_future'] if forecast else cls.combine_samples(samples, 'y')

    
    @classmethod
    def plot_samples(cls,
                     samples, 
                     plot_fields=['y'],
                     T=None, 
                     t=None, 
                     ax=None,                 
                     legend=True,
                     daily=False,
                     forecast=False):
        '''
        Plotting method for SIR-type models. 
        '''

        x = cls.get(samples, 'S')
        T_data = x.shape[1]
        if T is None or T > T_data:
            T = T_data

        fields = {f: cls.get(samples, f, daily=daily, forecast=forecast)[:,:T] for f in plot_fields}
        names = {f: cls.names[f] for f in plot_fields}

        medians = {names[f]: np.median(v, axis=0) for f, v in fields.items()}    
        pred_intervals = {names[f]: np.percentile(v, (10, 90), axis=0) for f, v in fields.items()}

        t = np.arange(T) if t is None else t[:T]

        ax = ax if ax is not None else plt.gca()

        ax.set_prop_cycle(None)

        # Plot medians
        df = pd.DataFrame(index=t, data=medians)
        df.plot(ax=ax, legend=legend)
        median_max = df.max().values

        # Plot prediction intervals
        pi_max = 10
        for pi in pred_intervals.values():
            ax.fill_between(t, pi[0,:], pi[1,:], alpha=0.1, label='CI')
            pi_max = np.maximum(pi_max, np.nanmax(pi[1,:]))

        return median_max, pi_max        
        
