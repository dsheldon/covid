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
                  det_prob_d)

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

            params = (np.append(beta,np.repeat(beta[-rw_use_last:].mean(),T_future+1)), 
                      sigma, 
                      gamma, 
                      forecast_rw_scale, 
                      drift, 
                      det_prob[-rw_use_last:].mean(),
                      confirmed_dispersion, 
                      death_dispersion,
                      death_prob, 
                      death_rate, 
                      det_prob_d)

            beta_f, det_rate_rw_f, x_f, y_f, z_f = self.dynamics(T+T_future+1, 
                                                                 params, 
                                                                 x0,
                                                                 suffix="_future",N=N)

            x = None#np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)

        return beta, x, y, z, det_prob, death_prob
    
    def simulate_renewal(self,x0,N,T,theta,CONV_WIDTH=40): 
         def Geometric0(mu):
            '''Geometric RV supported on 0,1,...'''
            p = 1/(1+mu)
            log_p = np.log(p)
            log_1_minus_p = np.log(1-p)
            def log_prob(k):
              return np.where(k >= 0, k * log_1_minus_p + log_p, -np.inf)
            return log_prob

         def Geometric1(mu):
             '''Geometric RV supported on 1,2,...'''
             p = 1/mu
             log_p = np.log(p)
             log_1_minus_p = np.log(1-p)
             def log_prob(k):
                 return np.where(k > 0, (k-1) * log_1_minus_p + log_p, -np.inf)
             return log_prob
         beta, sigma, gamma,death_prob,death_rate = theta
       
         # U = latent period
            # V = infectious period
         U_logp = Geometric0(1/sigma)
         V_logp = Geometric1(1/gamma)
         D_logp = Geometric0(1/death_rate)
            # For some reason this gives closest match to the diff eq. model
            # with U drawn from the geometric distribution supported on non-
            # negative integers and V drawn from the geometric supported on
            # positive integers.
            
         t = np.arange(CONV_WIDTH)
            
         U_pmf = np.exp(U_logp(t))
         U_ccdf = 1 - np.cumsum(U_pmf)
        
         V_pmf = np.exp(V_logp(t))
         V_ccdf = 1- np.cumsum(V_pmf)
            
         D_pmf = np.exp(D_logp(t))
            # A(t) = Pr(infectious t time units after being infected) 
            #      = sum_u Pr(U=u) * Pr(V >= t-u)
            #      = convolution of U pmf and V ccdf
            
         A = np.convolve(U_pmf, V_ccdf, mode='full')[:CONV_WIDTH]
         A_tmp = np.flip(np.convolve(U_pmf,V_pmf,mode='full'))[:CONV_WIDTH]
         A_rev = A[::-1] # to facilitate convolution incide the dynamics loop
            
            #print("R0", beta*A.sum())
            #print("beta/gamma", beta/gamma)
            
            # Let dE(t) be newly exposed cases at time t. Then
            #
            #  dE(t) = beta * S(t)/N * (# previous cases that are infectious at time t)
            #        = beta * S(t)/N * sum_{s<t} dE(s)*A(t-s)
            #        = beta * S(t)/N * conv(incidence, A)
            #
         def scan_body(state, beta):        
                incidence_history, S = state
                dE = beta * S/N * np.sum(incidence_history * A)
                new_state = (np.append(incidence_history[1:], dE), S-dE)
                return new_state, dE
         dE0 = x0 
         incidence_history = np.append(np.zeros(CONV_WIDTH-1), dE0)
         S = N - dE0
         _, dE = jax.lax.scan(scan_body, (incidence_history, S), beta*np.ones(T-1))
            
         dE = np.append(dE0, dE)
            
            # calculate other variables from incident exposures using 
            # various convolutions to "project forward" incident exposures
         E = np.convolve(dE, U_ccdf, mode='full')[:T]
         dI = np.convolve(dE, U_pmf, mode='full')[:T]
         I = np.convolve(dE, A, mode='full')[:T]
         dH = np.convolve(death_prob*dI, V_pmf, mode='full')[:T]
         dD = np.convolve(dH, D_pmf,mode='full')[:T]
          
         CE = np.cumsum(dE)
         CI = np.cumsum(dI)
         S = N - CE
            
         R = N-S-E-I
         return (dI,dD) 
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
        det_prob_d = params

        rw = frozen_random_walk("rw" + suffix,
                                num_steps=T-1,
                                num_frozen=num_frozen)
        
        beta = numpyro.deterministic("beta", beta0 * np.exp(rw_scale*rw))
        
        det_prob = numpyro.sample("det_prob" + suffix,
                                  LogisticRandomWalk(loc=det_prob0, 
                                                     scale=rw_scale, 
                                                     drift=0.,
                                                     num_steps=T-1))

        # Run ODE
        theta = (beta,sigma,gamma,death_prob,death_rate) 
        new_cases,new_deaths = self.simulate_renewal(x0[1],N,T,theta,CONV_WIDTH=40) 
        # Don't let incident cases/deaths be exactly zero (or worse, negative!)
        new_cases = np.maximum(new_cases[1:], 0.001)
        new_deaths = np.maximum(new_deaths[1:], 0.001)
 
        # Noisy observations
        with numpyro.handlers.scale(scale=0.5):
            if suffix != "_future":
                 y = observe_nb2("dy" + suffix, new_cases, det_prob, confirmed_dispersion, obs = confirmed)
            else:
                 y = observe_nb2("dy" + suffix, new_cases[-T_future:], det_prob[-T_future:], confirmed_dispersion, obs = confirmed)


        with numpyro.handlers.scale(scale=2.0):
            if suffix != "_future":
                z = observe_nb2("dz" + suffix, new_deaths, det_prob_d, death_dispersion, obs = death)  
            else:
                z = observe_nb2("dz" + suffix, new_deaths[-T_future:], det_prob_d, death_dispersion, obs = death)

        x=None
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
