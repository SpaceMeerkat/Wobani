#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:07:01 2017

@author: jamesdawson
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist


###############################################################################

### Define your kernel function. Here we use a Gaussian ###

def sec(a,b, length_scale , sigma) : 
    K = sigma * np.exp(-1/(2*length_scale) * cdist(a,b)**2)
    return K 

###############################################################################

def GP_prior(a , b, mu , kernel , length_scale, sigma , samples ) :
    f = np.random.multivariate_normal(mu.flat, kernel(a ,b , length_scale , sigma ) , samples)
    return f

def Posterior(introduced_err,x_train,ytrain,x_prior,length_scale,sigma,samples):

    K_train = sec(x_train , x_train , length_scale,sigma)
    K_prior = sec(x_prior , x_prior , length_scale,sigma)
    K_pt =  sec(x_prior , x_train , length_scale,sigma)
    K_tp = sec(x_train , x_prior ,  length_scale,sigma)  ## = k_tp transpose

    mean_function = np.dot(np.dot(K_pt ,np.linalg.inv((K_train)+((introduced_err**2)*np.eye(len(x_train))))  ), ytrain) 
    covariance_function = K_prior - np.dot(np.dot(K_pt ,np.linalg.inv((K_train) + ((introduced_err**2)*np.eye(len(x_train))))) , K_tp) 

    post = np.random.multivariate_normal(mean_function[:,0],covariance_function , samples)

    std = np.diag(covariance_function)
    
    return mean_function,covariance_function,post,std

###############################################################################

### Load in the relevant data #################################################
    
data = np.loadtxt('NGC0383_major_axis_cut.txt')

x = data[:,0]
y = data[:,1]
err = np.ones(len(x))*0.0013

FHWM = 0.15
width = FHWM/(2*np.sqrt(2*np.log(2)))

###############################################################################

### Define the prior space and sampling frequency #############################

x_prior = np.arange(-6,6,0.01)
x_prior = x_prior.reshape(-1,1)
mu = np.zeros(x_prior.shape)
prior_std = 1

###############################################################################

### Formulate and sample the prior space. Here we sample 5 possible functions #

prior = GP_prior(x_prior ,x_prior, mu , sec , width ,prior_std, 5)

plt.figure()
plt.grid()
plt.title('Samples from the Gaussian prior of instantiations of functions')
plt.plot(x_prior , prior.T,)
plt.show()
plt.savefig('prior sampling GP prior')

###############################################################################

### Define the training (seen data). This is our loaded in data ###############

x_train = x.reshape(-1,1)
ytrain = y.reshape(-1,1)
sigma = err.reshape(-1,1)

###############################################################################

### Formulate the posterior distribution. We sample 100 posterior functions ###

mean_function,covariance_function,post,std = Posterior(err,x_train,ytrain,x_prior,width,prior_std,100)

plt.figure()
for i in range(len(post)):
    plt.plot(x_prior,post[i,:],'g-',alpha=0.1)
    
plt.errorbar(x_train,ytrain, yerr=sigma,color='k')

plt.plot(x_prior , mean_function.flat, 'r--')
plt.plot(x_prior , mean_function.flat + (2*std), 'b-.')
plt.plot(x_prior , mean_function.flat - (2*std), 'b-.')

y1 = mean_function[:,0] - (2*std)
y2 = mean_function[:,0] - (2*std)

plt.xlabel('x') 
plt.ylabel('$f(x)$')
plt.grid()
plt.show()
plt.savefig('GP example noise')

###############################################################################

### Saving the output mean and standard deviation of the posterior ############

output_data = np.transpose(np.vstack([mean_function.flat,std]))
np.savetxt('Mean and 1 standard deviation GP posterior',output_data,delimiter=' ')

###############################################################################