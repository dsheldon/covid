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

         det_prob0 = numpyro.sample("det_prob0",
                                        dist.Beta(det_prob_est * det_prob_conc,
                                                 (1-det_prob_est) * det_prob_conc))
     
      
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
      
         E0 = numpyro.sample("E0", dist.Uniform(0, 1e-4*N))  # change to 1e-3 if starting on 2020-03-16
     
         rw = frozen_random_walk("rw" ,
                                     num_steps=T-1,
                                     num_frozen=num_frozen)
     
         beta0 = numpyro.sample("beta0",
                                    dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))
         beta = numpyro.deterministic("beta", beta0 * np.exp(rw_scale*rw))
         det_prob = numpyro.sample("det_prob" ,
                                       LogisticRandomWalk(loc=det_prob0,
                                                          scale=rw_scale,
                                                         drift=0.,
                                                          num_steps=T-1))
         
     
     
         if T_future >0:
             beta = np.concatenate((beta,np.repeat(beta[-10:].mean(),T_future)))
             det_prob =  np.concatenate((det_prob,np.repeat(det_prob[-10:].mean(),T_future)))
            
         
         death_dispersion = numpyro.sample("death_dispersion", 
                                                dist.TruncatedNormal(low=0.1,
                                                                     loc=death_dispersion, 
                                                                     scale=0.15))
     
         confirmed_dispersion = numpyro.sample("confirmed_dispersion", 
                                                   dist.TruncatedNormal(low=0.1,
                                                                        loc=confirmed_dispersion, 
                                                                        scale=0.15))
     
     
         sigma = numpyro.sample("sigma", 
                                    dist.Gamma(sigma_shape, sigma_shape * E_duration_est))
         gamma = numpyro.sample("gamma", 
                                     dist.Gamma(gamma_shape, gamma_shape * I_duration_est))
     
     
     
     
     
         det_prob_d = numpyro.sample("det_prob_d", 
                                         dist.Beta(.9 * 100,
                                                   (1-.9) * 100))
     
         death_prob = numpyro.sample("death_prob", 
                                         dist.Beta(death_prob_est * death_prob_conc, (1-death_prob_est) * death_prob_conc))
                                         
         death_rate = numpyro.sample("death_rate", 
                                         dist.Gamma(death_rate_shape, death_rate_shape * H_duration_est))
     
     
     
     
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
         
         def SEIR_renewal(theta, T=100, dE0=10, N=1e6, CONV_WIDTH=32):
     
              beta, sigma, gamma,death_prob,death_rate, pd = theta
     
         # U = latent period
         # V = infectious period
              U_logp = Geometric0(1/sigma)
           # need an infectious period 
              V_logp = Geometric0(1/gamma)
     
     
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
         #      = sum_u Pr(U=u) * [Pr(R >= t-u)*Pr(H >= t-u)]
         #      = convolution of U pmf and V ccdf
         
              A = np.convolve(U_pmf, V_ccdf, mode='full')[:CONV_WIDTH]
     
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
                 dE = beta * S/N * np.sum(incidence_history * A_rev)
                 new_state = (np.append(incidence_history[1:], dE), S-dE)
                 return new_state, dE
     
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
     
              dD = np.convolve(dH, D_pmf, mode='full')[:T]  
      
     
              CE = np.cumsum(dE)
              CI = np.cumsum(dI)
              S = N - CE
         
              R = N-S-E-I
              return (S, E, I, R,CI, dI, dD)
     
         pd= numpyro.sample('time_to_death',dist.Dirichlet(.1*np.ones(40)))
         theta = (beta, sigma, gamma,death_prob,death_rate,pd)
         if T_future == 0:
              S2, E2, I2, R2,CI, dI ,dD= SEIR_renewal(theta,T=T, dE0=E0, CONV_WIDTH=40)
         else:
              S2, E2, I2, R2,CI, dI ,dD= SEIR_renewal(theta,T=T_old+T_future, dE0=E0, CONV_WIDTH=40)
         
         if confirmed is not None:
             dy0 = numpyro.sample('dy0',dist.Normal(0,1),obs=confirmed[0])
             dz0 = numpyro.sample('dz0',dist.Normal(0,1),obs=death[0])
         else:
             dy0 = numpyro.sample('dy0',dist.Normal(0,1))
             dz0 = numpyro.sample('dz0',dist.Normal(0,1))
 
         y = dy0
         z = dz0 
         if confirmed is not None and T_future ==0:
             with numpyro.handlers.scale(scale=1.0):
                 y = observe_nb2("dy" , np.diff(CI), det_prob, confirmed_dispersion, obs = confirmed)
     
             with numpyro.handlers.scale(scale=1.0):
                 z = observe_nb2("dz" , dD[1:], det_prob_d, death_dispersion, obs = death)

         elif T_future > 0:
             with numpyro.handlers.scale(scale=1.0):
                 y = observe_nb2("dy_future" , np.diff(CI)[-28:], det_prob[-28:], confirmed_dispersion)
     
             with numpyro.handlers.scale(scale=1.0):
                 z = observe_nb2("dz_future" , dD[1:][-28:], det_prob_d, death_dispersion)
             with numpyro.handlers.scale(scale=1.0):
                 y = observe_nb2("dy" , np.diff(CI), det_prob, confirmed_dispersion)

             with numpyro.handlers.scale(scale=1.0):
                 z = observe_nb2("dz" , dD[1:], det_prob_d, death_dispersion)   

          # Sample initial number of infected individuals
         else:
             with numpyro.handlers.scale(scale=1.0):
                 y = observe_nb2("dy" , np.diff(CI), det_prob, confirmed_dispersion)

             with numpyro.handlers.scale(scale=1.0):
                 z = observe_nb2("dz" , dD[1:], det_prob_d, death_dispersion)
         y = np.append(dy0,y)
         z = np.append(dz0,z)
     
         return beta, None, y, z, det_prob, death_prob
         
    

    
    
    
    
    

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
