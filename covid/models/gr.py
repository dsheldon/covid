import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe,observe_normal, observe_nb2, ExponentialRandomWalk, LogisticRandomWalk, frozen_random_walk, clean_daily_obs
from .base import SEIRDBase, getter

import numpy as onp


"""
************************************************************
SEIRD model
************************************************************
"""

class SEIRD(SEIRDBase):    
    
    def __call__(self,
                 T = 50,
                 N = 1e5,
                 T_future = 0,
                 E_duration_est = 4.0,
                 I_duration_est = 2.0,
                 R0_est = 3.0,
                 beta_shape = 1.,
                 sigma_shape = 100.,
                 gamma_shape = 100.,
                 det_prob_est = 0.3,
                 det_prob_conc = 50.,
                 confirmed_dispersion=0.3,
                 death_dispersion=0.3,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0.,
                 drift_scale = None,
                 num_frozen=0,
                 rw_use_last=1,
                 confirmed=None,
                 death=None):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
                
        # Sample initial number of infected individuals


        # Sample dispersion parameters around specified values

        death_dispersion = numpyro.sample("death_dispersion", 
                                           dist.TruncatedNormal(low=0.1,
                                                                loc=death_dispersion, 
                                                                scale=0.15))




        rw = numpyro.sample("rw",dist.GaussianRandomWalk(scale=.1, num_steps=T))
        rw = (1+rw)
        beta0 = numpyro.sample("beta",
                               dist.Normal(1,1))
        D0 = numpyro.sample("D0", dist.Normal(.000002*N,1000))	
        sigma = numpyro.sample("sigma", 
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))
  
 
        gamma = numpyro.sample("gamma", 
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

       # Sample parameters
            
        if death is None:
            death = None
            death0 = None
        else: 
            death = clean_daily_obs(onp.diff(death))

            death0 = death[0]
         # First observation
        z_hat = np.exp(np.cumsum(np.log(beta0*rw)) + np.log(D0))

        z0 = observe_normal("dz0",z_hat[0] , .95, death_dispersion, obs = death0)
        y0 = observe_normal("dy0",z_hat[0] , .95, death_dispersion, obs = death0) 
        z = observe_normal("dz",np.diff(z_hat) , .95, death_dispersion, obs = death)  
        y = observe_normal("dy",np.diff(z_hat) , .95, death_dispersion, obs = death)
        z= np.append(z0,z)
        y=np.append(y0,y)
        if (T_future >0):

            z_hat_future =np.exp(np.cumsum(np.log(beta0*np.repeat(rw[-1],T_future))) + np.log(D0)) 

            z_future = observe_normal("dz_future",np.diff(z_hat_future) , .95, death_dispersion, obs = death)

            y_future = observe_normal("dy_future",np.diff(z_hat_future) , .95, death_dispersion, obs = death)
            z = np.append(z,z_future)
            y = np.append(y,y_future)

        return beta0, None, y,z, None, None

    
    
    
    
    

    dy = getter('dy')
    dz = getter('dz')
    
    def y0(self, **args):
        return self.z0(**args)

    
    def y(self, samples, **args):
        '''Get cumulative cases from incident ones'''
        
        dy = self.dy(samples, **args)
        
        y0 = np.zeros(dy.shape[0])
        if args.get('forecast'):
            y0 = self.y(samples, forecast=False)[:,-1]
 
        return y0[:,None] + onp.cumsum(dy, axis=1)


    def z0(self, **args):
        return self.z0(**args)

    
    def z(self, samples, **args):
        '''Get cumulative deaths from incident ones'''
        
        dz = self.dz(samples, **args)
        
        z0 = np.zeros(dz.shape[0])
        if args.get('forecast'):
            z0 = self.z(samples, forecast=False)[:,-1]
 
        return z0[:,None] + onp.cumsum(dz, axis=1)
