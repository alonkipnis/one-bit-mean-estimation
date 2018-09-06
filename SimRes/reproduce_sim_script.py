
# CJ has its own randState upon calling
# to reproduce results one needs to set
# the internal State of the global stream
# to the one saved when ruuning the code for
# the fist time;
import os,sys,pickle,numpy,random;
CJsavedState = pickle.load(open('CJrandState.pickle','rb'));
numpy.random.set_state(CJsavedState['numpy_CJsavedState']);
random.setstate(CJsavedState['CJsavedState']);
    
sys.path.append('.');
from aux import *
import numpy as np

nMonte = 10000

n = 1000

Amax = inv_Phi(0.95)
Amin = -inv_Phi(0.95)

err_var_split = np.zeros(n+1)
err_split = np.zeros(n+1)
err_var_grad = np.zeros(n+1)
err_grad = np.zeros(n+1)
err_var_bayes = np.zeros(n+1)
err_bayes = np.zeros(n+1)
err_var_sdm = np.zeros(n+1)
err_sdm = np.zeros(n+1)
err_var_smp = np.zeros(n)
err_smp = np.zeros(n)

sig = 1

for i in range(nMonte) :
    th = Amin + (Amax-Amin)*np.random.rand()# uniform prior over (Amin,Amax)
    z = sig * np.random.randn(n)
    y = z + th
    
    #Split
    split_ratio = np.sqrt(n) / n
    th_hat_split = [split_est(y[:k], split_ratio = split_ratio, th0 = 6*(np.random.rand()-0.5), inv_PDF = inv_Phi) for k in range(n+1)]
    err_split += th-np.array(th_hat_split)
    err_var_split += (th-np.array(th_hat_split))**2
    
    #SGD
    th_hat_grad = grad_descent(y, g0 = 0.68)
    err_grad += th-np.array(th_hat_grad)
    err_var_grad += (th-np.array(th_hat_grad))**2
    
    #Bayes
    th_hat_bayes = bayes_adaptive(y, Amin, Amax)
    err_bayes += th-np.array(th_hat_bayes)
    err_var_bayes += (th-np.array(th_hat_bayes))**2
    
    #th_hat_sdm = sigma_delta(y,v0  = 0)
    #err_sdm += th-np.array(th_hat_sdm)
    #err_var_sdm += (th-np.array(th_hat_sdm))**2
    
    #Sample mean
    mean_hat = np.cumsum(y) / np.arange(1,n+1)
    err_smp += th - mean_hat
    err_var_smp += (th-np.array(mean_hat))**2


err_var_split /= nMonte
err_split /= nMonte
err_var_grad /= nMonte
err_grad /= nMonte
err_var_bayes /= nMonte
err_bayes /= nMonte    
#err_var_sdm /= nMonte
#err_sdm /= nMonte    
err_var_smp /= nMonte
err_smp /= nMonte    

print("Done!")

np.savetxt('err_var_split_n{}_nMonte{}_lmd{}.csv'.format(n,nMonte,split_ratio),err_var_split, fmt='%.8f', delimiter=',')
np.savetxt('err_var_grad_{}_nMonte{}.csv'.format(n,nMonte),err_var_grad, fmt='%.8f', delimiter=',')
np.savetxt('err_var_bayes_{}_nMonte{}.csv'.format(n,nMonte),err_var_bayes, fmt='%.8f', delimiter=',')
#np.save('err_var_sdm_{}_nMonte{}'.format(n,nMonte),err_var_sdm)