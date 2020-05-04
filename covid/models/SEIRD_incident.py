import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, ExponentialRandomWalk
from .base import SEIRDBase, getter

import numpy as onp

"""
************************************************************
SEIRD model
************************************************************
"""

class SEIRD_incident(SEIRDBase):

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
                 det_noise_scale = 0.5,
                 rw_scale = 1e-1,
                 drift_scale = None,
                 confirmed = None,
                 death=None):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''

        # Sample initial number of infected individuals
        I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
        E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
        H0 = numpyro.sample("H0", dist.Uniform(0, 0.02*N))
        D0 = numpyro.sample("D0", dist.Uniform(0, 1000))

        # Sample parameters
        sigma = numpyro.sample("sigma", 
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))

        gamma = numpyro.sample("gamma", 
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))

    #     gamma = numpyro.sample("gamma", 
    #                            dist.TruncatedNormal(loc = 1./I_duration_est, scale = 0.25)

        beta0 = numpyro.sample("beta0", 
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

        det_prob = numpyro.sample("det_prob", 
                                  dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))

        det_prob_d = numpyro.sample("det_prob_d", 
                                    dist.Beta(.9 * 100,
                                              (1-.9) * 100))

        death_prob = numpyro.sample("death_prob", 
                                    dist.Beta(.1 * 100,
                                              (1-.1) * 100))

        death_rate = numpyro.sample("death_rate", 
                                    dist.Gamma(10, 10 * 10))

        if drift_scale is not None:
            drift = numpyro.sample("drift", dist.Normal(loc=0, scale=drift_scale))
        else:
            drift = 0


        x0 = SEIRDModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        confirmed0, confirmed = (None, None) if confirmed is None else (confirmed[0], np.diff(confirmed))
        death0, death = (None, None) if death is None else (death[0], np.diff(death))


        # First observation
        with numpyro.handlers.scale(scale_factor=0.5):
            y0 = observe("dy0", x0[6], det_prob, det_noise_scale, obs=confirmed0)

        with numpyro.handlers.scale(scale_factor=2.0):
            z0 = observe("dz0", x0[5], det_prob_d, det_noise_scale, obs=death0)


        params = (beta0, sigma, gamma, 
                  rw_scale, drift, 
                  det_prob, det_noise_scale, 
                  death_prob, death_rate, det_prob_d)

        beta, x, y, z = self.dynamics(T, params, x0, 
                                       confirmed = confirmed,
                                       death = death)

        x = np.vstack((x0, x))
        y = np.append(y0, y)
        z = np.append(z0, z)

        if T_future > 0:

            params = (beta[-1], sigma, gamma, 
                      rw_scale, drift, 
                      det_prob, det_noise_scale, 
                      death_prob, death_rate, det_prob_d)

            beta_f, x_f, y_f, z_f = self.dynamics(T_future+1, params, x[-1,:], 
                                                  suffix="_future")

            x = np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)

        return beta, x, y, z, det_prob, death_prob

    
    def dynamics(self, T, params, x0, confirmed=None, death=None, suffix=""):
        '''Run SEIRD dynamics for T time steps'''

        beta0, sigma, gamma, rw_scale, drift, \
        det_prob, det_noise_scale, death_prob, death_rate, det_prob_d  = params

        beta = numpyro.sample("beta" + suffix,
                      ExponentialRandomWalk(loc=beta0, scale=rw_scale, drift=drift, num_steps=T-1))

        # Run ODE
        x = SEIRDModel.run(T, x0, (beta, sigma, gamma, death_prob, death_rate))

        numpyro.deterministic("x" + suffix, x[1:])

        x_diff = np.diff(x, axis=0)

        # Noisy observations
        with numpyro.handlers.scale(scale_factor=0.5):
            y = observe("dy" + suffix, x_diff[:,6], det_prob, det_noise_scale, obs = confirmed)   

        with numpyro.handlers.scale(scale_factor=2.0):
            z = observe("dz" + suffix, x_diff[:,5], det_prob_d, det_noise_scale, obs = death)  

        return beta, x, y, z


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
