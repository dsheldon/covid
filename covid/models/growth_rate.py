import jax
import jax.numpy as np
from jax.random import PRNGKey
from ..glm import glm, GLM, log_link, logit_link, Gamma, Beta
from functools import partial

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import get_future_data,observe, observe_nb2, observe_normal, ExponentialRandomWalk, LogisticRandomWalk, frozen_random_walk, clean_daily_obs
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



        #exp_coef =  numpyro.sample("exp_coef",dist.Normal(0,100))
        #exp_coef_d =  numpyro.sample("exp_coef_d",dist.Normal(0,100))

        #rw_scale=.005
        #rw_add =  numpyro.sample("rw_add",dist.GaussianRandomWalk(scale=rw_scale, num_steps=T))

        #rw2_add =  numpyro.sample("rw2_add",dist.GaussianRandomWalk(scale=rw_scale, num_steps=T))
        #rw_add = rw_add + exp_coef
        #rw2_add = rw2_add +exp_coef_d
        exp_coef_int =  numpyro.sample("exp_coef_int",dist.Normal(0,100))
        exp_coef_d_int =  numpyro.sample("exp_coef_d_int",dist.Normal(0,100))
        
        R0_glm_conf = GLM("1 + state_of_emergency + shelter_in_place + Q('non-contact_school')", 
                 place_data, 
                 log_link,
                 partial(Gamma, var=1),
                 prior = dist.Normal(0, 1),
                 guess=.2,
                 name="R0")
        R0_conf = R0_glm_conf.sample(shape=(-1))[0]
        R0_glm_d = GLM("1 + state_of_emergency + shelter_in_place + Q('non-contact_school') + cr(t,df=3)",
                 place_data,
                 log_link,
                 partial(Gamma, var=1),
                 prior = dist.Normal(0, 1),
                 guess=.2,
                 name="R0d")
        R0_d = R0_glm_d.sample(shape=(-1))[0]
        confirmed_hat =R0_conf#rw_add*np.arange(T)
        death_hat = R0_d #rw2_add*np.arange(T)

        # First observation
        var_c = numpyro.sample("var_c",dist.Normal(0,10))
        var_d = numpyro.sample("var_d",dist.Normal(0,10))
        var_c = np.exp(var_c)
        var_d = np.exp(var_d)
        with numpyro.handlers.scale(scale_factor=1.0):
            y0 = numpyro.sample("y0" , dist.Normal(confirmed_hat[0], var_c),obs = confirmed0)
            numpyro.deterministic("mean_y0"  , y0)
        with numpyro.handlers.scale(scale_factor=1.0):
            z0 = numpyro.sample("z0" , dist.Normal(death_hat[0],var_d ), obs = death0)
            numpyro.deterministic("mean_z0"  , z0)

        with numpyro.handlers.scale(scale_factor=1.0):
            y = numpyro.sample("y" , dist.Normal(confirmed_hat[1:], var_c),obs = confirmed)
            numpyro.deterministic("mean_y"  , y)
        with numpyro.handlers.scale(scale_factor=1.0):
            z = numpyro.sample("z" , dist.Normal(death_hat[1:],var_d ), obs = death)        
            numpyro.deterministic("mean_z"  , z)
        y = np.append(y0, y)
        z = np.append(z0, z)
        if T_future > 0:
       
             future_data = get_future_data(place_data, T_future-1)
             confirmed_hat_future = R0_glm_conf.sample(future_data, name="R0_future", shape=(-1))[0] 
             death_hat_future = R0_glm_d.sample(future_data, name="R0_future_d", shape=(-1))[0]

             with numpyro.handlers.scale(scale_factor=1.0):
                 y_f = numpyro.sample("y"+"_future" , dist.Normal(confirmed_hat_future, var_c))
                 numpyro.deterministic("mean_y_future"  ,y_f)
             with numpyro.handlers.scale(scale_factor=1.0):
                 z_f = numpyro.sample("z" +"_future" , dist.Normal(death_hat_future,var_d)) 
                 numpyro.deterministic("mean_z_future"  , z_f)
             y = np.append(y, y_f)
             z = np.append(z, z_f)

        return  None,None,np.exp(y), np.exp(z)
