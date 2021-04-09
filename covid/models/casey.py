import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

from ..compartment import SEIRDModel
from .util import observe, observe_nb2, ExponentialRandomWalk, LogisticRandomWalk, frozen_random_walk, clean_daily_obs
from .base import SEIRDBase, getter
from jax.experimental.ode import odeint
from jax.scipy.special import expit, xlog1py, xlogy, gammaln
import numpy as onp
import math

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
                 death=None,
                 T_old=0):

        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''        
        def NB2(mu=None, k=None):
            conc = 1./k
            rate = conc/mu
            return dist.GammaPoisson(conc, rate)
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
  

        def fill_nan(A):
             '''
             interpolate to fill nan values
             '''
             inds = onp.arange(A.shape[0])
             good = onp.where(onp.isfinite(A))
             f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
             B  = onp.where(onp.isfinite(A),A,f(inds))
             return B

        if confirmed is not None:
           confirmed = clean_daily_obs(onp.diff(confirmed))
           death = clean_daily_obs(onp.diff(death))
        basis = splines.bs(onp.arange(0,T+28),knots=knots,degree=1)

#preds=predict.predict(basis, newx=np.arange(len(df_full.X.values),len(df_full.X.values)+28))
#pred_mat = np.array(preds)
        
        basis_matrix = onp.array(basis)
        num_basis = basis_matrix.shape[1]
        if (T_future == 0):
            basis_train= basis_matrix[:T,:]

            basis_oos_matrix =basis_train#
        else:
            basis_oos_matrix=basis_matrix
        det_prob0 = numpyro.sample("det_prob0",
                                   dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))        



        tau = numpyro.sample('scale_on_coef',dist.Normal(0,1000))
        a_raw = numpyro.sample('a_raw',dist.GaussianRandomWalk(scale=1e-2, num_steps=num_basis))        
        
        if confirmed is not None:
             y0 = numpyro.sample('dy0',dist.Normal(0,1),obs=confirmed[0])
             z0 = numpyro.sample('dz0',dist.Normal(0,1),obs=death[0])        
        else:
             y0 = numpyro.sample('dy0',dist.Normal(0,1))
             z0 = numpyro.sample('dz0',dist.Normal(0,1))
        case_detection = numpyro.sample('cfr_rw',LogisticRandomWalk(det_prob0,scale=.5,num_steps=T))

        if T_future > 0:
           case_detection = np.concatenate((case_detection,np.repeat(case_detection[-1],28)))
        #   beta = np.concatenate((beta,np.repeat(beta[-1],28)))
        case_detection = numpyro.deterministic('det_prob',case_detection)
  
        
#        tau = numpyro.sample('scale_on_coef',dist.Normal(0,100))

        


        y_hat =  numpyro.deterministic('y_hat', np.exp(np.dot(np.array(basis_oos_matrix), np.dot(tau,a_raw))))

        cfr = numpyro.sample('cfr',dist.Beta(1,100))

        par=0
        offset = 40
        prior_mean = .1*onp.ones(offset)
        p = numpyro.sample('time_to_death',dist.Dirichlet(prior_mean))
   
        if T_future ==0:
           num_data = 0
        else:
           num_data = T_old +T_future

        def pdf(par,t):
            return p[t]

        #d_hat = numpyro.deterministic('d_hat',cfr*np.convolve(y_hat,(pdf_evaluated))[:T])
        
        d_hat = numpyro.deterministic('d_hat',cfr*( pdf(par,39)*y_hat[:num_data-offset] +
pdf(par,38)*y_hat[1:num_data-offset+1] +
pdf(par,37)*y_hat[2:num_data-offset+2] +
pdf(par,36)*y_hat[3:num_data-offset+3]+
pdf(par,35)*y_hat[4:num_data-offset+4]+
pdf(par,34)*y_hat[5:num_data-offset+5] +
pdf(par,33)*y_hat[6:num_data-offset+6]+
pdf(par,32)*y_hat[7:num_data-offset+7]+
pdf(par,31)*y_hat[8:num_data-offset+8] +
pdf(par,30)*y_hat[9:num_data-offset+9] +
pdf(par,29)*y_hat[10:num_data-offset+10] +
pdf(par,28)*y_hat[11:num_data-offset+11] +
pdf(par,27)*y_hat[12:num_data-offset+12] +
pdf(par,26)*y_hat[13:num_data-offset+13] +
pdf(par,25)*y_hat[14:num_data-offset+14] +
pdf(par,24)*y_hat[15:num_data-offset+15]+
pdf(par,23)*y_hat[16:num_data-offset+16] +
pdf(par,22)*y_hat[17:num_data-offset+17] +
pdf(par,21)*y_hat[18:num_data-offset+18] +
pdf(par,20)*y_hat[19:num_data-offset+19] +
pdf(par,19)*y_hat[20:num_data-offset+20] +
pdf(par,18)*y_hat[21:num_data-offset+21] +
pdf(par,17)*y_hat[22:num_data-offset+22] +
pdf(par,16)*y_hat[23:num_data-offset+23] +
pdf(par,15)*y_hat[24:num_data-offset+24]+
pdf(par,14)*y_hat[25:num_data-offset+25] +
pdf(par,13)*y_hat[26:num_data-offset+26]+
pdf(par,12)*y_hat[27:num_data-offset+27] +
pdf(par,11)*y_hat[28:num_data-offset+28] +
pdf(par,10)*y_hat[29:num_data-offset+29] +
pdf(par,9)*y_hat[30:num_data-offset+30] +
pdf(par,8)*y_hat[31:num_data-offset+31] +
pdf(par,7)*y_hat[32:num_data-offset+32] +
pdf(par,6)*y_hat[33:num_data-offset+33] +
pdf(par,5)*y_hat[34:num_data-offset+34] +
pdf(par,4)*y_hat[35:num_data-offset+35]+
pdf(par,3)*y_hat[36:num_data-offset+36] +
pdf(par,2)*y_hat[37:num_data-offset+37]+
pdf(par,1)*y_hat[38:num_data-offset+38] +
pdf(par,0)*y_hat[39:num_data-offset+39]  ))






        death_dispersion = numpyro.sample("death_dispersion", 
                                            dist.TruncatedNormal(low=0.1,
                                                                loc=death_dispersion, 
                                                                scale=0.15))


        confirmed_dispersion = numpyro.sample("confirmed_dispersion", 
                                              dist.TruncatedNormal(low=0.1,
                                                                   loc=confirmed_dispersion, 
                                                                   scale=0.15))



        print (case_detection.shape)
        print (y_hat.shape)
        print (confirmed)

        if confirmed is not None:
            y =numpyro.sample('dy', NB2(case_detection*y_hat, confirmed_dispersion), obs=confirmed)
            z =numpyro.sample('dz', NB2(d_hat, death_dispersion), obs=death[offset:])
              
        else:
            y =numpyro.sample('dy', NB2(case_detection*y_hat, confirmed_dispersion))        
            z =numpyro.sample('dz', NB2(np.concatenate((np.zeros(offset),d_hat)), death_dispersion))
        
        if T_future > 0:
             z_f =  numpyro.sample('dz_future', NB2(np.concatenate((np.zeros(offset),d_hat))[-T_future:], death_dispersion))##), obs=death  )
             y_f = numpyro.sample('dy_future', NB2((case_detection*y_hat)[-T_future:], confirmed_dispersion))#, obs=confirmed)
             z = np.append(z,z_f)
             y= np.append(y,y_f)
        # Sample parameters

        y = np.append(y0,y)
        z = np.append(z0,z)
        return None, None, y, z, None, None
    
    

    
    
    
    
    
    

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
