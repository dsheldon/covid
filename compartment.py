import abc
import jax
from jax.experimental.ode import build_odeint
import jax.numpy as np


'''
Class to represent dynamics and simulation of compartment models.
Does not store parameters or state values. These are always passed in
'''
class CompartmentModel(object):
    
    @abc.abstractmethod
    def dx_dt(self, x, theta):
        '''Compute time derivative'''
        return
        
    def __init__(self, rtol=1e-5, atol=1e-3, mxstep=500):
        
        self.odeint = build_odeint(self.dx_dt, 
                                   rtol=rtol,
                                   atol=atol, 
                                   mxstep=mxstep)
    
    def run(self, T, x0, theta):
        
        # Theta is a tuple of parameters. Entries are 
        # scalars or vectors of length T-1
        is_scalar = [np.ndim(a)==0 for a in theta]
        if np.all(is_scalar):
            return self.run_static(T, x0, theta)        
        else:
            return self.run_time_varying(T, x0, theta)
            
    def run_static(self, T, x0, theta):
        t = np.arange(T, dtype='float32')
        return self.odeint(x0, t, theta)

    def run_time_varying(self, T, x0, theta):
        theta = tuple(np.broadcast_to(a, (T-1,)) for a in theta)
        
        t_one_step = np.array([0.0, 1.0])
        
        def advance(x0, theta):
            x1 = self.odeint(x0, t_one_step, theta)[1]
            return x1, x1

        # Run T–1 steps of the dynamics starting from the intial distribution
        _, X = jax.lax.scan(advance, x0, theta, T-1)
        return np.vstack((x0, X))
        

class SIRModel(CompartmentModel):

    def dx_dt(self, x, t, theta):
        """
        SIR equations
        """        
        S, I, R, C = x
        beta, gamma = theta
        N = S + I + R
        
        dS_dt = - beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        dC_dt = beta * S * I / N  # cumulative infections
        
        return np.stack([dS_dt, dI_dt, dR_dt, dC_dt])

    @staticmethod
    def R0(theta):
        beta, gamma = theta
        return beta/gamma
    
    @staticmethod
    def growth_rate(theta):
        beta, gamma = theta
        return beta - gamma

    @staticmethod
    def seed(N=1e6, I=100.):
        '''
        Seed infection. Return state vector for I infected out of N
        '''
        return np.array([N-I, I, 0.0, I])
        
class SEIRModel(CompartmentModel):
    
    def dx_dt(self, x, t, theta):
        """
        SEIR equations
        """        
        S, E, I, R, C = x
        beta, sigma, gamma = theta
        N = S + E + I + R
        
        dS_dt = - beta * S * I / N
        dE_dt = beta * S * I / N - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I
        dC_dt = sigma * E  # cumulative infections
        
        return np.stack([dS_dt, dE_dt, dI_dt, dR_dt, dC_dt])

    @staticmethod
    def R0(theta):
        beta, sigma, gamma = theta
        return beta / gamma
    
    @staticmethod
    def growth_rate(theta):
        '''
        Initial rate of exponential growth
        
        Reference: Junling Ma, Estimating epidemic exponential growth rate 
        and basic reproduction number, Infectious Disease Modeling, 2020
        '''
        beta, sigma, gamma = theta
        return (-(sigma + gamma) + np.sqrt((sigma - gamma)**2 + 4 * sigma * beta))/2.
        
    @staticmethod
    def seed(N=1e6, I=100., E=0.):
        '''
        Seed infection. Return state vector for I exponsed out of N
        '''
        return np.array([N-E-I, E, I, 0.0, I])
