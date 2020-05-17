import jax
import jax.numpy as np
from jax.random import PRNGKey
from jax.config import config; config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, observe_nb2,observe_normal, ExponentialRandomWalk, LogisticRandomWalk
from .base import SEIRDBase, getter

import numpy as onp


def frozen_random_walk(name, num_steps=100, num_frozen=10):

    # last random value is repeated frozen-1 times
    num_random = min(max(0, num_steps - num_frozen), num_steps)
    num_frozen = num_steps - num_random

    rw = numpyro.sample(name, dist.GaussianRandomWalk(num_steps=num_random))
    rw = np.concatenate((rw, np.repeat(rw[-1], num_frozen)))    
    return observe_normal,rw

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
                 R0_est = 4,
                 beta_shape = 500,
                 sigma_shape = 5,
                 gamma_shape = 8,
                 det_prob_est = 0.3,
                 det_prob_conc = 50,
                 confirmed_dispersion_est=0.5,
                 death_dispersion_est=0.5,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0.0,
                 drift_scale = None,
                 num_frozen=0,
                 confirmed=None,
                 death=None,
                 num_places=1,
	          cov_dat=None):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
        #Hack right now to avoid major refactor
        confirmed_and_death = confirmed
        num_places = onp.uint32(num_places)
        
        print (cov_dat)
        sys.exit()

        beta_intervention_total =  numpyro.sample("beta_total",
                               dist.Normal(0,.5))
        with numpyro.plate("num_places", num_places): 
  
      # Sample initial number of infected individuals
             I0 = numpyro.sample("I0", dist.Uniform(0.0, 0.002*N))
             E0 = numpyro.sample("E0", dist.Uniform(0.0, 0.002*N))
             H0 = numpyro.sample("H0", dist.Uniform(0.0, 1e-5*N))
             D0 = numpyro.sample("D0", dist.Uniform(0.0, 1e-5*N))


        # Sample dispersion parameters around specified values
             confirmed_dispersion = numpyro.sample("confirmed_dispersion", 
                                              dist.TruncatedNormal(low=0.0,
                                                                   loc=confirmed_dispersion_est, 
                                                                   scale=confirmed_dispersion_est))


             death_dispersion = numpyro.sample("death_dispersion", 
                                           dist.TruncatedNormal(low=0.0,
                                                                loc=death_dispersion_est, 
                                                                scale=death_dispersion_est))

        
        # Sample parameters
             beta0 = numpyro.sample("beta0",
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))
             
             betasp = numpyro.sample("betasp",
                               dist.Normal(beta_intervention_total,.1))

             det_prob_d = numpyro.sample("det_prob_d", 
                                    dist.Beta(.9 * 100.0,
                                              (1-.9) * 100.0))

             death_prob = numpyro.sample("death_prob", 
                                    dist.Beta(.015 * 1000.0,
                                              (1-.015) * 1000.0))

             death_rate = numpyro.sample("death_rate", 
                                    dist.Gamma(1000.0, 1000.0 * 10.0))

             if drift_scale is not None:
                 drift = numpyro.sample("drift", dist.Normal(loc=0.0, scale=drift_scale))
             else:
                 drift = 0.0
        det_prob= .15
        gamma = np.repeat(1.0/I_duration_est,num_places).astype(np.float32)
        sigma = np.repeat(1.0/E_duration_est,num_places).astype(np.float32)
        #beta0 = np.repeat(3.5,num_places)       
  #x0 = SEIRDModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
        x0 = jax.vmap(SEIRDModel.seed)(N, I0, E0, H0, D0)
        numpyro.deterministic("x0", x0)
        NoneType = type(None)
        if isinstance(confirmed,NoneType) == False:
           use_obs = True
        else:
           use_obs = False
        # Split observations into first and rest
        if use_obs:
            confirmed= confirmed_and_death[:,:,0]
            death = confirmed_and_death[:,:,1]

            confirmed0, confirmed = confirmed[:,0], np.diff(confirmed[:,1:],axis=1)
            death0, death = death[:,0], np.diff(death[:,1:],axis=1)
        else:
            confirmed0,confirmed = None, None
            death0, death = None, None 
        # First observation
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = observe_normal("dy0", x0[:,6], 0.1, confirmed_dispersion, obs=confirmed0)
            
        with numpyro.handlers.scale(scale_factor=2.0):
            z0 = observe_normal("dz0", x0[:,5], det_prob_d, death_dispersion, obs=death0)

        params = (beta0, 
                  sigma, 
                  gamma, 
                  rw_scale, 
                  drift, 
                  .5, 
                  confirmed_dispersion, 
                  death_dispersion,
                  death_prob, 
                  death_rate, 
                  det_prob_d,betasp)

        beta, det_prob, x, y, z = self.dynamics(T, 
                                                params, 
                             	                   x0,
                                                num_frozen = num_frozen,
                                                confirmed = confirmed,
                                                death = death,
                                                 num_places = num_places)

        x = np.concatenate((x0[:,None,:], x), axis=1)
        y = np.concatenate((y0[:,None], y), axis=1)
        z = np.concatenate((z0[:,None], z), axis=1)
        if T_future > 0:

            params = (beta[:,-1], 
                      sigma, 
                      gamma, 
                      forecast_rw_scale, 
                      drift, 
                      det_prob,#det_prob[:,-1], 
                      confirmed_dispersion, 
                      death_dispersion,
                      death_prob, 
                      death_rate, 
                      det_prob_d,
                      betasp)

            beta_f, det_rate_rw_f, x_f, y_f, z_f = self.dynamics(T_future+1, 
                                                                 params, 
                                                                 x[:,-1,:], num_places = num_places,
                                                                 suffix="_future")

            x = np.concatenate((x, x_f), axis=1)
            y = np.concatenate((y, y_f), axis=1)
            z = np.concatenate((z, z_f), axis=1)

        return beta, x, y, z, det_prob, death_prob
    
    
    def dynamics(self, T, params, x0, num_frozen=0, confirmed=None, death=None, num_places=1, suffix=""):
        '''Run SEIRD dynamics for T time steps'''

        beta0, \
        sigma, \
        gamma, \
        rw_scale, \
        drift, \
        det_prob0, \
        confirmed_dispersion, \
        death_dispersion, \
        death_prob, \
        death_rate, \
        det_prob_d,\
        betasp = params
        
        det_prob = .15
        sigma = sigma[:,None]
        gamma = gamma[:,None]
        death_rate = death_rate[:,None]
        death_prob = death_prob[:,None]
        det_prob_d = det_prob_d[:,None]
        confirmed_dispersion = confirmed_dispersion[:,None]
        death_dispersion = death_dispersion[:,None]
        
        with numpyro.plate("places", num_places):
            rw = numpyro.sample("rw" + suffix,
                                ExponentialRandomWalk(loc = 1.0,
                                                      scale = 1e-1,
                                                      drift = 0.0, 
                                                      num_steps = T-1))  
            #det_prob_phase_1 = numpyro.sample("det_prob_phase_1"+suffix,dist.Beta(.1 * 1000.0,
            #                                  (1-.1) * 1000.0))
            #det_prob_phase_2 = numpyro.sample("det_prob_phase_2"+suffix,dist.Beta(.2 * 1000.0,
             #                                 (1-.2) * 1000.0))
            #det_prob_phase_3 = numpyro.sample("det_prob_phase_3"+suffix,dist.Beta(.4 * 1000.0,
            #                                  (1-.4) * 1000.0))

        #det_prob = np.concatenate([ np.repeat(det_prob_phase_1[:,None],7,axis=1) ,np.repeat(det_prob_phase_2[:,None],7,axis=1),np.repeat(det_prob_phase_3[:,None],T-2-14,axis=1)] ,axis=1)
        
        covariates = onp.zeros((num_places,T-1))
        covariates[0,10:] = 1.
        covariates[1,15:] = 1.	
        
        beta = beta0[:,None]* rw 
        if suffix != "_future":
             beta = beta*np.exp(betasp[:,None]*covariates)
        apply_model = lambda x0, beta, sigma, gamma, death_prob, death_rate: SEIRDModel.run(T, x0, (beta, sigma, gamma, death_prob, death_rate))
        x = jax.vmap(apply_model)(x0, beta, sigma, gamma, death_prob, death_rate)
        x = x[:,1:,:] # drop first time step from result (duplicates initial value)
        # Run ODE
        x_diff = np.diff(x, axis=1)
        # Noisy observations
        # need to drop one det_prob because of differencing 
        with numpyro.handlers.scale(scale_factor=0.5):
            y = observe_normal("dy" + suffix, x_diff[:,:,6], .15,confirmed_dispersion , obs = confirmed)   

        with numpyro.handlers.scale(scale_factor=2.0):
            z = observe_normal("dz" + suffix, x_diff[:,:,5], det_prob_d, death_dispersion, obs = death)  

        
        return beta, det_prob, x, y, z

    
    
    
    
    

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
