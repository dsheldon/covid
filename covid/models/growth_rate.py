import jax
import jax.numpy as np
from jax.random import PRNGKey
from ..glm import glm, GLM, log_link, logit_link, Gamma, Beta
from functools import partial

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, observe_nb2, observe_normal, ExponentialRandomWalk, LogisticRandomWalk, frozen_random_walk, clean_daily_obs
from .base import SEIRDBase, getter

import numpy as onp



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
                 confirmed_dispersion=0.15,
                 death_dispersion=0.15,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0.,
                 drift_scale = None,
                 num_frozen=0,
                 rw_use_last=1,
                 confirmed=None,
                 death=None,
                 place_data=None):

        # Split observations into first and rest
        if confirmed is None:
            confirmed0, confirmed = (None, None)
        else:
            confirmed=np.log(confirmed)
            confirmed0 = confirmed[0]
            confirmed = confirmed[1:]
            
        if death is None:
            death0, death = (None, None)
        else: 
            death = np.log(death)
            death0 = death[0]
            death = death[1:]


        # exponential growth model 
 
        rw_add =  numpyro.sample("rw_add",dist.GaussianRandomWalk(scale=100, num_steps=T))
        rw2_add =  numpyro.sample("rw2_add",dist.GaussianRandomWalk(scale=100, num_steps=T))

        exp_coef =  numpyro.sample("exp_coef",dist.Normal(0,1))
        exp_coef_d =  numpyro.sample("exp_coef_d",dist.Normal(0,1))
   

        exp_coef_int =  numpyro.sample("exp_coef_int",dist.Normal(0,1))
        exp_coef_d_int =  numpyro.sample("exp_coef_d_int",dist.Normal(0,1))

        confirmed_hat = exp_coef_int + exp_coef*np.arange(T)
        death_hat = exp_coef_d_int + exp_coef_d*np.arange(T)

        # First observation
        var_c = numpyro.sample("var_c",dist.Gamma(1,1))
        var_d = numpyro.sample("var_d",dist.Gamma(1,1))
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = numpyro.sample("y0" , dist.Normal(confirmed_hat[0], var_c),obs = confirmed0)
            numpyro.deterministic("mean_y0"  , y0)
        with numpyro.handlers.scale(scale_factor=2.0):
            z0 = numpyro.sample("z0" , dist.Normal(death_hat[0],var_d ), obs = death0)
            numpyro.deterministic("mean_z0"  , z0)

        with numpyro.handlers.scale(scale_factor=0.5):
            y = numpyro.sample("y" , dist.Normal(confirmed_hat[1:], var_c),obs = confirmed)
            numpyro.deterministic("mean_y"  , y)
        with numpyro.handlers.scale(scale_factor=2.0):
            z = numpyro.sample("z" , dist.Normal(death_hat[1:],var_d ), obs = death)        
            numpyro.deterministic("mean_z"  , z)
        y = np.append(y0, y)
        z = np.append(z0, z)
        if T_future > 0:
        
             confirmed_hat = exp_coef_int + exp_coef*np.arange(T,T_future)
             death_hat = exp_coef_d_int + exp_coef_d*np.arange(T,T_future)
             with numpyro.handlers.scale(scale_factor=0.5):
                 y_f = numpyro.sample("y"+"_future" , dist.Normal(confirmed_hat[1:], 1),obs = confirmed)
                 numpyro.deterministic("mean_y_future"  ,y_f)
             with numpyro.handlers.scale(scale_factor=2.0):
                 z_f = numpyro.sample("z" +"_future" , dist.Normal(death_hat[1:],1 ), obs = death) 
                 numpyro.deterministic("mean_z_future"  , z_f)
             y = np.append(y, y_f)
             z = np.append(z, z_f)

        return  None,None,np.exp(y), np.exp(z)
