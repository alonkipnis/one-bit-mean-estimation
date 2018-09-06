#gradient descent implementation
from scipy.special import erf
from scipy.stats import norm
import numpy as np


def median(prob,x_vals) :
    return x_vals[prob.cumsum()>=0.5][0]

def sign(x) :
    return 2*(x>0)-1

def grad_descent(y, th0 = 0, g0 = 0.75) :
    #must be  2/3 < g0 < 1
    n = y.shape[0]
    th_t = th0
    th_bar = 0
    th_hat = []
    th_hat.append(th0)
    for i in range(n) :
        th_t = th_t + (i+1)**(-g0)*sign(y[i]-th_t)
        th_bar = th_bar + (th_t-th_bar) / (i+1)
        th_hat.append(th_bar)
    return th_hat
    
def Qfunc(x) :
    return 0.5*(1-erf(x/np.sqrt(2)))

def Phi(x) :
    return norm.cdf(x)

def inv_Phi(x, sig = 1) :
    return norm.ppf(x, loc=0, scale = sig)

def phi(x) :
    return norm.pdf(x)

def bayes_adaptive(y, Amin, Amax, th_prior = None, xx = None, eps = 1e-3) :

    n = y.shape[0]
    
    if th_prior == None : 
        #uniform prior
        dx = (Amax-Amin)*eps/np.sqrt(n)
        xx = np.arange(Amin,Amax,dx)
        th_prior = (xx<Amax)*(xx>Amin)*1.0/(Amax-Amin)*dx
        #least favorable
        #th_prior = (np.exp(-(xx-Amax) ** 2 * 10.) + np.exp(-(xx+Amin) ** 2 * 10. ))
    
    th_all = []
    th_prior = th_prior/th_prior.sum()
    th_hat = (xx*th_prior).sum()
    th_all.append(th_hat) #zero information estimation
    for i in range(n) :
        #th_hat = median(th_prior,xx)
        
        M = 2*((y[i]-th_hat)>0)-1
        th_prior = th_prior * Qfunc(M*(th_hat-xx))
        th_prior = th_prior/th_prior.sum()
        th_hat = (xx*th_prior).sum()
        
        th_all.append(th_hat)
    return th_all
    
def binary_entropy(x) :
    return -x * np.log2(x)-(1-x) * np.log2(1-x)

#two sample estimator 

def split_est(x, split_ratio = 0.5, th0 = 0, inv_PDF = inv_Phi) : 
    n = len(x)
    n1 = int(split_ratio*n+0.5)
    
    x1 = x[:n1]
    x2 = x[n1:]

    th_hat2 = np.nan
    
    if(n1 > 0 and n-n1 > 0) :
        #first step
        th0 = 0
        M1 = (x1 > th0) + .0 #one bit messages
        pn1 = M1.mean()
        pn1 = (np.arctan(2*(pn1 - 0.5))+1)/2
        th_hat1 = th0 + inv_PDF(pn1)
        #print("diff1 = {}".format(th_hat1-th))

        #second step
        M2 = (x2 > th_hat1) + .0 #one bit messages
        pn2 = M2.mean()
        pn2 = (np.arctan(2*(pn2 - 0.5))+1)/2
        th_hat2 = th_hat1 + inv_PDF(pn2)

    return th_hat2

#Sigma-Delta Estimator

def sigma_delta(y, v0 = 0,gain = 1) :
    n = y.shape[0]
    qs = []
    v = v0
    q = 0
    th_hat = np.zeros_like(y)
    for i in range(n) :
        v = v - q + y[i]
        m = (2 * (v>0) - 1)
        q = gain * m
        qs.append(q)
        th_hat[i] = np.mean(qs)
    return th_hat