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





        rw = numpyro.sample("rw",dist.GaussianRandomWalk(scale=100, num_steps=T))
        D0 = numpyro.sample("D0", dist.Normal(.000002*N,1000))	

       # Sample parameters
            
        if death is None:
            death = None
            death0 = None
        else: 
            death = clean_daily_obs(death)

            death0 = death[0]
         # First observation
        z_hat = np.exp(np.cumsum(rw) + np.log(D0))

        z0 = observe_normal("z0",z_hat[0] , .95, death_dispersion, obs = death0)
        y0 = observe_normal("y0",z_hat[0] , .95, death_dispersion, obs = death0) 
        z = observe_normal("z",z_hat , .95, death_dispersion, obs = death)  
        y = observe_normal("y",z_hat , .95, death_dispersion, obs = death)
        z= np.append(z0,z)
        y=np.append(y0,y)
        if (T_future >0):

            z_hat_future =np.exp(np.cumsum(np.repeat(rw[-1],T_future)) + np.log(D0)) 

            z_future = observe_normal("z_future",z_hat_future , .95, death_dispersion, obs = death)

            y_future = observe_normal("y_future",z_hat_future , .95, death_dispersion, obs = death)
            z = np.append(z,z_future)
            y = np.append(y,y_future)

        return None, None, y,z, None, None

    
    
    
    
    

