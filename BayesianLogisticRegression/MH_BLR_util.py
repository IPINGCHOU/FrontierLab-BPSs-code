# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:39:49 2019

@Title: FrontierLab exchange program - Metropolis-Hastings source code (for Bayesian Logistic Regression)
@Author: Chou I-Ping
"""
import numpy as np
import copy
import time
from scipy.stats import norm

def expit(z):
    return np.exp(z) / (1 + np.exp(z))

class MH:
    def __init__(self, X, Y, b_prior_sd):
        self.X = X
        self.Y = Y
        self.all_samples = []
        self.b_prior_sd = b_prior_sd
        self.success = X[np.where(Y == 1)]
        self.failure = X[np.where(Y == 0)]
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        
        self.beta = np.random.normal(0,1,2)
        self.p = 0
        
    def log_post(self,beta):
        success_prob = expit(beta[0] + beta[1] * self.success)
        failure_prob = expit(beta[0] + beta[1] * self.failure)
        log_success = np.sum(np.log(success_prob))
        log_failure = np.sum(np.log(1-failure_prob))
        log_prior = np.sum(np.log(norm.pdf(beta, 0, self.b_prior_sd)))
        return log_success + log_failure + log_prior
        
    
    def sampler(self, can_sd, burninIters, iterations, store_skip, verbose):
        self.all_samples.append(self.beta)
        cur_lp = self.log_post(self.beta)
        
        for i in range(1,int(iterations),1):
            can_beta = copy.deepcopy(self.beta)
            for j in range(2):
                can_beta[j] = np.random.normal(self.beta[j], can_sd,1)
            can_lp = self.log_post(can_beta)
            R = np.exp(can_lp - cur_lp)
            U = np.random.uniform(0,1,1)

            if U<R:
                self.beta = can_beta
                cur_lp = can_lp
                self.p += 1
                    
            if(i % store_skip == 0):
                self.all_samples.append(copy.deepcopy(self.beta))
            if(i % verbose == 0):
                print('Current process: ' + str(i))
                
            if(i == burninIters):
                self.burnin_sample = copy.deepcopy(i)
                self.burnin_time = time.time()
            
        self.all_samples = np.array(self.all_samples)
                
                

                
    
