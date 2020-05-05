import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt


'''Utility to define access method for time varying fields'''
def getter(f):
    def get(self, samples, forecast=False):
        return samples[f + '_future'] if forecast else self.combine_samples(samples, f)
    return get


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
        'z': 'deaths',
        'dy': 'daily confirmed',
        'dz': 'daily deaths'
    }
            
    
    def __init__(self, data=None, mcmc_samples=None, **args):
        self.mcmc_samples = mcmc_samples
        self.data = data
        self.args = args

        
    @property
    def obs(self):
        '''Provide extra arguments for observations
        
        Used during inference and forecasting
        '''
        return {}
    

    """
    ***************************************
    Inference and sampling routines
    ***************************************
    """
    
    def infer(self, num_warmup=1000, num_samples=1000, num_chains=1, rng_key=PRNGKey(1), **args):
        '''Fit using MCMC'''
        
        args = dict(self.args, **args)
        
        kernel = NUTS(self, init_strategy = numpyro.infer.util.init_to_median())

        mcmc = MCMC(kernel, 
                    num_warmup=num_warmup, 
                    num_samples=num_samples, 
                    num_chains=num_chains)
             
        mcmc.run(rng_key, **self.obs, **args)    
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

        args = dict(self.args, **args)
        return predictive(rng_key, **self.obs, **args)
        
    
    """
    ***************************************
    Data access and plotting
    ***************************************
    """    
    
    def combine_samples(self, samples, f, use_future=False):
        '''Combine fields like x0, x, x_future into a single array'''
        
        f0, f_future = f + '0', f + '_future'
        data = np.concatenate((samples[f0][:,None], samples[f]), axis=1)
        if f_future in samples and use_future:
            data = np.concatenate((data, samples[f_future]), axis=1)
        return data
    
    
    def get(self, samples, c, **kwargs):
        
        forecast = kwargs.get('forecast', False)
        
        if c in self.compartments:
            x = samples['x_future'] if forecast else self.combine_samples(samples, 'x')
            j = self.compartments.index(c)
            return x[:,:,j]
        
        else:
            return getattr(self, c)(samples, **kwargs)  # call method named c
        

    def horizon(self, samples, **kwargs):
        '''Get time horizon'''
        y = self.y(samples, **kwargs)
        return y.shape[1]
        
        
    '''These are methods e.g., call self.z(samples) to get z'''
    z = getter('z')
    y = getter('y')
    mean_y = getter('mean_y')
    mean_z = getter('mean_z')

    # There are only available in some models but easier to define here
    dz = getter('dz')
    dy = getter('dy')
    mean_dy = getter('mean_dy')
    mean_dz = getter('mean_dz')
    
    
    def plot_samples(self,
                     samples, 
                     plot_fields=['y'],
                     start='2020-03-04',
                     T=None,
                     ax=None,          
                     legend=True,
                     forecast=False):
        '''
        Plotting method for SIR-type models. 
        '''

        ax = plt.axes(ax)

        T_data = self.horizon(samples, forecast=forecast)        
        T = T_data if T is None else min(T, T_data) 
        
        fields = {f: self.get(samples, f, forecast=forecast)[:,:T] for f in plot_fields}
        names = {f: self.names[f] for f in plot_fields}
                
        medians = {names[f]: np.median(v, axis=0) for f, v in fields.items()}    
        pred_intervals = {names[f]: np.percentile(v, (10, 90), axis=0) for f, v in fields.items()}

        t = pd.date_range(start=start, periods=T, freq='D')

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
    
    
    def plot_forecast(self,
                      variable,
                      post_pred_samples, 
                      forecast_samples=None,
                      start='2020-03-04',
                      T_future=7*4,
                      ax=None,
                      obs=None,
                      scale='lin'):

        ax = plt.axes(ax)
        
        # Plot posterior predictive for observed times
        self.plot_samples(post_pred_samples, ax=ax, start=start, plot_fields=[variable])
                
        # Plot forecast
        T = self.horizon(post_pred_samples)
        obs_end = pd.to_datetime(start) + pd.Timedelta(T-1, "d")
        forecast_start = obs_end + pd.Timedelta("1d")
        
        median_max, pi_max = self.plot_samples(forecast_samples,
                                               start=forecast_start,
                                               T=T_future,
                                               ax=ax,
                                               forecast=True,
                                               legend=False,
                                               plot_fields=[variable])
        
        # Plot observation
        forecast_end = forecast_start + pd.Timedelta(T_future-1, "d")
        obs[start:forecast_end].plot(ax=ax, style='o')
        
        # Plot vertical line at end of observed data
        ax.axvline(obs_end, linestyle='--', alpha=0.5)
        ax.grid(axis='y')
        
        
        # Scaling and axis limits
        if scale == 'log':
            ax.set_yscale('log')

            # Don't display below 1
            bottom, top = ax.get_ylim()
            bottom = 1 if bottom < 1 else bottom
            ax.set_ylim([bottom, pi_max])
        else:
            top = np.minimum(2*median_max, pi_max)
            ax.set_ylim([0, top])

        
        return median_max, pi_max
    
    
    
class SEIRDBase(Model):

    compartments = ['S', 'E', 'I', 'R', 'H', 'D', 'C']

    @property
    def obs(self):
        '''Provide extra arguments for observations
        
        Used during inference and forecasting
        '''        
        if self.data is None:
            return {}

        return {
            'confirmed': self.data['confirmed'].values,
            'death': self.data['death'].values
           }
    
    
    def dz_mean(self, samples, **args):
        '''Daily deaths mean'''
        mean_z = self.mean_z(samples, **args)
        if args.get('forecast'):
            first = self.mean_z(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan
            
        return onp.diff(mean_z, axis=1, prepend=first)        
    
    def dz(self, samples, noise_scale=0.4, **args):
        '''Daily deaths with observation noise'''
        dz_mean = self.dz_mean(samples, **args)
        dz = dist.Normal(dz_mean, noise_scale * dz_mean).sample(PRNGKey(10))
        return dz
        
    def dy_mean(self, samples, **args):
        '''Daily confirmed cases mean'''
        mean_y = self.mean_y(samples, **args)
        if args.get('forecast'):
            first = self.mean_y(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan
            
        return onp.diff(mean_y, axis=1, prepend=first)
    
    def dy(self, samples, noise_scale=0.4, **args):
        '''Daily confirmed cases with observation noise'''
        dy_mean = self.dy_mean(samples, **args)
        dy = dist.Normal(dy_mean, noise_scale * dy_mean).sample(PRNGKey(11))
        return dy
