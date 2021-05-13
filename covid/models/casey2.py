import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, observe_nb2, ExponentialRandomWalk, LogisticRandomWalk, frozen_random_walk, clean_daily_obs
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
                 H_duration_est = 10.0,
                 R0_est = 3.0,
                 beta_shape = 1.,
                 sigma_shape = 100.,
                 gamma_shape = 100.,
                 death_rate_shape = 10.,
                 det_prob_est = 0.3,
                 det_prob_conc = 50.,
                 confirmed_dispersion=0.3,
                 death_dispersion=0.3,
                 rw_scale = 2e-1,
                 death_prob_est=0.01,
                 death_prob_conc=100,
                 forecast_rw_scale = 0.,
                 drift_scale = None,
                 num_frozen=0,
                 rw_use_last=1,
                 confirmed=None,
                 death=None,
                  T_old=None):


        import rpy2
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        from scipy import interpolate

        rpy2.robjects.numpy2ri.activate()
        splines = importr("splines")
        predict = importr("stats")
        T=T-1
        knots=onp.arange(0,T-10,10)
        knots = onp.concatenate((knots,onp.array([T+28])))
        basis = splines.bs(onp.arange(0,T+28),knots=knots,degree=1)
        basis_matrix = onp.array(basis)
        num_basis = basis_matrix.shape[1]


        tau = numpyro.sample('scale_on_coef',dist.Normal(0,1000))
        a_raw = numpyro.sample('a_raw',dist.GaussianRandomWalk(scale=1e-2, num_steps=num_basis))
        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
                
        # Sample initial number of infected individuals
        I0 = numpyro.sample("I0", dist.Uniform(0, 1e-4*N))  # change to 1e-3 if starting on 2020-03-16
        E0 = numpyro.sample("E0", dist.Uniform(0, 1e-4*N))  # change to 1e-3 if starting on 2020-03-16
        H0 = numpyro.sample("H0", dist.Uniform(0, 1e-4*N))
        D0 = numpyro.sample("D0", dist.Uniform(0, 1e-4*N))


        # Sample dispersion parameters around specified values

        death_dispersion = numpyro.sample("death_dispersion", 
                                           dist.TruncatedNormal(low=0.1,
                                                                loc=death_dispersion, 
                                                                scale=0.15))


        confirmed_dispersion = numpyro.sample("confirmed_dispersion", 
                                              dist.TruncatedNormal(low=0.1,
                                                                   loc=confirmed_dispersion, 
                                                                   scale=0.15))



        
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
                                    dist.Beta(death_prob_est * death_prob_conc, (1-death_prob_est) * death_prob_conc))
                                    
        death_rate = numpyro.sample("death_rate", 
                                    dist.Gamma(death_rate_shape, death_rate_shape * H_duration_est))

        if drift_scale is not None:
            drift = numpyro.sample("drift", dist.Normal(loc=0., scale=drift_scale))
        else:
            drift = 0.

        death_dist = numpyro.sample("death_dist",dist.Dirichlet(1*np.ones(40)))
        x0 = SEIRDModel.seed(N=N, I=I0, E=E0, H=H0, D=D0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        if confirmed is None:
            confirmed0, confirmed = (None, None)
        else:
            confirmed0 = confirmed[0]
            confirmed = clean_daily_obs(onp.diff(confirmed))
            
        if death is None:
            death0, death = (None, None)
        else: 
            death0 = death[0]
            death = clean_daily_obs(onp.diff(death))
        
        # First observation
        with numpyro.handlers.scale(scale=0.5):
            y0 = observe_nb2("dy0", x0[6], det_prob0, confirmed_dispersion, obs=confirmed0)
            
        with numpyro.handlers.scale(scale=2.0):
            z0 = observe_nb2("dz0", x0[5], det_prob_d, death_dispersion, obs=death0)

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
                  det_prob_d, 
                  death_dist,
                  basis,tau,a_raw)
        beta, det_prob, x, y, z = self.dynamics(T, 
                                                params, 
                                                x0,
                                                num_frozen = num_frozen,
                                                confirmed = confirmed,
                                                death = death,
                                                 N=N)

        x = None#np.vstack((x0, x))
        y = np.append(y0, y)
        z = np.append(z0, z)

        if T_future > 0:

            params = (beta0, 
                      sigma, 
                      gamma, 
                      forecast_rw_scale, 
                      drift, 
                      det_prob[-rw_use_last:].mean(),
                      confirmed_dispersion, 
                      death_dispersion,
                      death_prob, 
                      death_rate, 
                      det_prob_d,
                      death_dist,basis,tau,a_raw)

            beta_f, det_rate_rw_f, x_f, y_f, z_f = self.dynamics(T+T_future+1, 
                                                                 params, 
                                                                 x0,
                                                                 suffix="_future",N=N)

            x = None#np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)

        return beta, x, y, z, det_prob, death_prob
    
    def dynamics(self, T, params, x0, num_frozen=0, confirmed=None, death=None, suffix="",N=None):
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
        det_prob_d, \
        death_dist,basis,tau,a_raw = params

        rw = frozen_random_walk("rw" + suffix,
                                num_steps=T-1,
                                num_frozen=num_frozen)
        
        beta = numpyro.deterministic("beta", beta0 * np.exp(rw_scale*rw))
        
        det_prob = numpyro.sample("det_prob" + suffix,
                                  LogisticRandomWalk(loc=det_prob0, 
                                                     scale=rw_scale, 
                                                     drift=0.,
                                                     num_steps=T))

        # Run ODE
        def Geometric1(mu):
             '''Geometric RV supported on 1,2,...'''
             p = 1/mu
             log_p = np.log(p)
             log_1_minus_p = np.log(1-p)
             def log_prob(k):
                 return np.where(k > 0, (k-1) * log_1_minus_p + log_p, -np.inf)
             return log_prob 
        t = np.arange(40)
        V_logp = Geometric1(1/gamma)
        D_logp = Geometric1(1/death_rate)
        V_pmf = np.exp(V_logp(t))
        D_pmf = np.exp(D_logp(t))
        U_pmf =    np.exp(Geometric(1/sigma))    

        basis_matrix = onp.array(basis)
        num_basis = basis_matrix.shape[1]
        if (suffix != "_future"):
            basis_train= basis_matrix[:T,:]
            basis_oos_matrix =basis_train#
        else:
            basis_oos_matrix=basis_matrix


      

        new_cases = np.exp(np.dot(basis_oos_matrix,tau*a_raw))# Don't let incident cases/deaths be exactly zero (or worse, negative!)
        dI = np.convolve(new_cases,U_pmf,mode='full')[:T]
        dH = np.convolve(death_prob*dI, V_pmf, mode='full')[:T]
        new_deaths =  np.convolve(dH, D_pmf,mode='full')[:T]

        new_cases = np.maximum(dI, 0.001)
        new_deaths = np.maximum(new_deaths, 0.001)
        # Noisy observations
        with numpyro.handlers.scale(scale=0.5):
            if suffix != "_future":
                 y = observe_nb2("dy" + suffix, new_cases, det_prob, confirmed_dispersion, obs = confirmed)
            else:
                 y = observe_nb2("dy" + suffix, new_cases[-28:], det_prob[-28:], confirmed_dispersion, obs = confirmed)


        with numpyro.handlers.scale(scale=2.0):
            if suffix != "_future":
                z = observe_nb2("dz" + suffix, new_deaths, det_prob_d, death_dispersion, obs = death)
            else:
                z = observe_nb2("dz" + suffix, new_deaths[-28:], det_prob_d, death_dispersion, obs = death)
        return None, det_prob, None, y, z

    
    
    
    
    

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
