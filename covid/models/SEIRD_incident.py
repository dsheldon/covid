import jax
import jax.numpy as np
from jax.random import PRNGKey

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
                 R0_est = 3.0,
                 beta_shape = 1,
                 sigma_shape = 5,
                 gamma_shape = 8,
                 det_prob_est = 0.3,
                 det_prob_conc = 50,
                 confirmed_dispersion_est=0.15,
                 death_dispersion_est=0.15,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0,
                 drift_scale = None,
                 num_frozen=0,
                 confirmed=None,
                 death=None):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
        #Hack right now to avoid major refactor
        confirmed= np.transpose(confirmed[0,:,:])
        death = np.transpose(death[1,:,:])
        num_places = confirmed.shape[0]
        print (confirmed)
        print (death)
        with numpyro.plate("num_places", num_places): 
  
      # Sample initial number of infected individuals
             I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
             E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
             H0 = numpyro.sample("H0", dist.Uniform(0, 1e-3*N))
             D0 = numpyro.sample("D0", dist.Uniform(0, 1e-3*N))


        # Sample dispersion parameters around specified values
             confirmed_dispersion = numpyro.sample("confirmed_dispersion", 
                                              dist.TruncatedNormal(low=0.,
                                                                   loc=confirmed_dispersion_est, 
                                                                   scale=confirmed_dispersion_est))


             death_dispersion = numpyro.sample("death_dispersion", 
                                           dist.TruncatedNormal(low=0.,
                                                                loc=death_dispersion_est, 
                                                                scale=death_dispersion_est))

        
        # Sample parameters
             sigma = numpyro.sample("sigma", 
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))

             gamma = numpyro.sample("gamma", 
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))


             beta0 = numpyro.sample("beta0",
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

             det_prob0 = numpyro.sample("det_prob0", 
                                   dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))

             det_prob_d = numpyro.sample("det_prob_d", 
                                    dist.Beta(.9 * 100,
                                              (1-.9) * 100))

             death_prob = numpyro.sample("death_prob", 
                                    dist.Beta(.01 * 100,
                                              (1-.01) * 100))

             death_rate = numpyro.sample("death_rate", 
                                    dist.Gamma(10, 10 * 10))

             if drift_scale is not None:
                 drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
             else:
                 drift = 0


        #x0 = SEIRDModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
        x0 = jax.vmap(SEIRDModel.seed)(N, I0, E0, H0, D0)
        numpyro.deterministic("x0", x0)
        if confirmed is not None:
           use_obs = True
        else:
           use_obs = False
        # Split observations into first and rest
        if use_obs:
            confirmed0, confirmed = confirmed[:,0], np.diff(confirmed[:,1:],axis=1)
            death0, death = death[:,0], np.diff(death[:,1:],axis=1)
        else:
            obs0, obs = None, None
            death0, death = None, None 
        # First observation
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = observe_nb2("dy0", x0[:,6], det_prob0, confirmed_dispersion, obs=confirmed0)
            
        with numpyro.handlers.scale(scale_factor=2.0):
            z0 = observe_nb2("dz0", x0[:,5], det_prob_d, death_dispersion, obs=death0)

        params = (beta0, 
                  sigma, 
                  gamma, 
                  rw_scale, 
                  drift, 
                  det_prob0, 
                  confirmed_dispersion, 
                  death_dispersion,
                  death_prob, 
                  death_rate, 
                  det_prob_d)

        beta, det_prob, x, y, z = self.dynamics(T, 
                                                params, 
                             	                   x0,
                                                num_frozen = num_frozen,
                                                confirmed = confirmed,
                                                death = death)

        x = np.concatenate((x0[:,None,:], x), axis=1)
        y = np.concatenate((y0[:,None], y), axis=1)
        z = np.concatenate((z0[:,None], z), axis=1)
        if T_future > 0:

            params = (beta[:,-1], 
                      sigma, 
                      gamma, 
                      forecast_rw_scale, 
                      drift, 
                      det_prob[:,-1], 
                      confirmed_dispersion, 
                      death_dispersion,
                      death_prob, 
                      death_rate, 
                      det_prob_d)

            beta_f, det_rate_rw_f, x_f, y_f, z_f = self.dynamics(T_future+1, 
                                                                 params, 
                                                                 x[:,-1,:],
                                                                 suffix="_future")

            x = np.concatenate((x, x_f), axis=1)
            y = np.concatenate((y, y_f), axis=1)
            z = np.concatenate((z, z_f), axis=1)

        return beta, x, y, z, det_prob, death_prob
    
    
    def dynamics(self, T, params, x0, num_frozen=0, confirmed=None, death=None, suffix=""):
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
        det_prob_d = params

        num_places= beta0.shape
        sigma = sigma[:,None]
        gamma = gamma[:,None]
        death_rate = death_rate[:,None]
        death_prob = death_prob[:,None]
        det_prob_d = det_prob_d[:,None]
        confirmed_dispersion = confirmed_dispersion[:,None]
        death_dispersion = death_dispersion[:,None]
        
        with numpyro.plate("places", 2):
            rw = numpyro.sample("rw" + suffix,
                                ExponentialRandomWalk(loc = 1.0,
                                                      scale = rw_scale,
                                                      drift = 0., 
                                                      num_steps = T-1))  
            det_prob = numpyro.sample("det_prob" + suffix,
                                  LogisticRandomWalk(loc=0.3, 
                                                     scale=rw_scale, 
                                                     drift=0,
                                                     num_steps=T-2))
        beta = beta0[:,None]* rw
        apply_model = lambda x0, beta, sigma, gamma, death_prob, death_rate: SEIRDModel.run(T, x0, (beta, sigma, gamma, death_prob, death_rate))
        x = jax.vmap(apply_model)(x0, beta, sigma, gamma, death_prob, death_rate)
        x = x[:,1:,:] # drop first time step from result (duplicates initial value)
        # Run ODE
        x_diff = np.diff(x, axis=1)
        # Noisy observations
        # need to drop one det_prob because of differencing 
        with numpyro.handlers.scale(scale_factor=0.5):
            y = observe_normal("dy" + suffix, x_diff[:,:,6], det_prob, .15, obs = confirmed)   

        with numpyro.handlers.scale(scale_factor=2.0):
            z = observe_normal("dz" + suffix, x_diff[:,:,5], det_prob_d, .15, obs = death)  

        
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
