import jax
import jax.numpy as np
from jax.scipy.special import expit, logit

import numpyro
import numpyro.distributions as dist

import patsy

class OneHot(object):
    '''Overcomplete one-hot encoding for categorical variables'''
    def __init__(self, reference=0):
        self.reference = reference

    def code_with_intercept(self, levels):
        return patsy.ContrastMatrix(np.eye(len(levels)),
                                    ["[My.%s]" % (level,) for level in levels])

    def code_without_intercept(self, levels):
        return self.code_with_intercept(levels)

def log_link():
    return np.log, np.exp

def logit_link():
    return jax.scipy.special.logit, jax.scipy.special.expit

def Gamma(mu=None, var=None, dispersion=None):
    '''Return Gamma distribution with specified mean'''
    assert var is not None or dispersion is not None
    shape = mu**2/var if var is not None else 1./dispersion    
    rate = shape/mu
    return dist.Gamma(shape, rate)

def Beta(mu=None, conc=None):
    return dist.Beta(mu*conc, (1-mu)*conc)


class GLM(object):
    '''Class to sample coefficients and responses form Bayesian GLM'''
    
    def __init__(self, 
                 formula, 
                 data, 
                 link, 
                 family, 
                 prior=None, 
                 term_priors=None, 
                 guess=None, 
                 theta=None,
                 name="y"):
        
        self.X = patsy.dmatrix(formula, data)
        self.link = link
        self.family = family
        self.name = name
        self.guess = guess
                            
        if theta is None:
            '''Sample theta from prior'''
            _, d = self.X.shape
            
            theta = np.zeros(d)

            info = self.X.design_info
            column_names = info.column_names

            if term_priors is None:
                prior = prior if prior is not None else dist.Cauchy(0, 1)
                term_priors = [prior] * len(info.terms)

            for term, prior in zip(info.terms, term_priors):
                term_slice = info.term_slices[term]
                num_cols = len(column_names[term_slice])
                theta_term = numpyro.sample(name + "_" + term.name(), prior, sample_shape=(num_cols,))
                theta = jax.ops.index_update(theta, term_slice, theta_term)

        self.theta = theta 
            
        
    def sample(self, data=None, name=None, shape=None, obs=None):
        '''Sample responses'''
        name = name or self.name
        
        if data is None:
            X = self.X         # use same data used to create model
        else:
            info = self.X.design_info # information from original data
            X = patsy.dmatrix(info, data)   # design matrix for new data
        
        linpred = np.array(X) @ self.theta

        if shape is not None:
            linpred = linpred.reshape(shape)  # reshape to tensor if requested

        fwd, inv = self.link()

        if self.guess is None:
            mu = inv(linpred)
        else:
            fwd_guess = fwd(self.guess)
            if not np.isfinite(fwd_guess):
                raise ValueError("Bad Guess")
            mu = inv(fwd_guess + linpred)

        y = numpyro.sample(name, self.family(mu), obs=obs)
        
        return y, mu, linpred


def glm(*args, shape=None, obs=None, **kwargs):
    '''Sample coefficients and response variables from Bayesian GLM'''
    model = GLM(*args, **kwargs)
    return model.sample() + (model.X, model.theta)
    
