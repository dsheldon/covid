import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, ExponentialRandomWalk, LogisticRandomWalk
from .base import SEIRDBase

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
                 beta_shape = 1,
                 sigma_shape = 5,
                 gamma_shape = 8,
                 det_prob_est = 0.3,
                 det_prob_conc = 50,
                 det_noise_scale = 0.15,
                 rw_scale = 2e-1,
                 forecast_rw_scale = 0,
                 drift_scale = None,
                 confirmed=None,
                 death=None):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
                
        # Sample initial number of infected individuals
        I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
        E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
        H0 = numpyro.sample("H0", dist.Uniform(0, 0.02*N))
        D0 = numpyro.sample("D0", dist.Uniform(0, 0.02*N))

        # Sample parameters
        sigma = numpyro.sample("sigma", 
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))

        gamma = numpyro.sample("gamma", 
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))


        beta0 = numpyro.sample("beta0", 
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

        det_prob = numpyro.sample("det_prob", 
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


        x0 = SEIRDModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        confirmed0, confirmed = (None, None) if confirmed is None else (confirmed[0], confirmed[1:])
        death0, death = (None, None) if death is None else (death[0], death[1:])


        # First observation
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = observe("y0", x0[6], det_prob, det_noise_scale, obs=confirmed0)
        z0 = observe("z0", x0[5], det_prob_d, det_noise_scale, obs=death0)


        params = (beta0, sigma, gamma, 
                  rw_scale, drift, 
                  det_prob, det_noise_scale, 
                  death_prob, death_rate, det_prob_d,None)

        beta, det_prob_rw, x, y, z = self.dynamics(T, params, x0, 
                                       confirmed = confirmed, 
                                       death = death,
                                      det_rate=det_prob_est)

        x = np.vstack((x0, x))
        y = np.append(y0, y)
        z = np.append(z0, z)

        if T_future > 0:

            params = (beta[-1], sigma, gamma, 
                      rw_scale, drift, 
                      det_prob_rw[-1], det_noise_scale, 
                     death_prob, death_rate, det_prob_d,det_prob_rw[-1])

            beta_f, det_rate_rw_f, x_f, y_f, z_f = self.dynamics(T_future+1, params, x[-1,:], 
                                                   suffix="_future",det_rate=det_prob_est)

            x = np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)

        return beta, x, y, z, det_prob, death_prob
    
    def dynamics(self, T, params, x0, confirmed=None, death=None, det_rate=None, suffix=""):
        '''Run SEIRD dynamics for T time steps'''

        beta0, sigma, gamma, rw_scale, drift, \
        det_prob, det_noise_scale, death_prob, death_rate, det_prob_d,det_prob_future  = params

        beta = numpyro.sample("beta" + suffix,
                      ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))


        if suffix != "_future":
            det_prob_rw = numpyro.sample("det_rate_rw" + suffix,
                      LogisticRandomWalk(loc=beta0, scale=rw_scale, drift=0, num_steps=T-1))
        else:
            det_prob_rw = det_prob_future

        # Run ODE
        x = SEIRDModel.run(T, x0, (beta, sigma, gamma, death_prob, death_rate))
        x = x[1:] # first entry duplicates x0
        numpyro.deterministic("x" + suffix, x)


        # Noisy observations
        with numpyro.handlers.scale(scale_factor=0.5):
            y = observe("y" + suffix, x[:,6], det_prob_rw, det_noise_scale, obs = confirmed)

        with numpyro.handlers.scale(scale_factor=2.0):
            z = observe("z" + suffix, x[:,5], det_prob_d, det_noise_scale, obs = death)

        return beta,det_prob_rw, x, y, z


        