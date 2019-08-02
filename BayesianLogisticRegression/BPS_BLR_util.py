# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:12:07 2019

@Title: FrontierLab exchange program - BPS source code (for Bayesian Logistic Regression)
@Author: Chou I-Ping
@Reference: A. Bouchard-Ct, S. J. Vollmer, and A. Doucet, “The bouncy particle sampler: A non-reversible rejection-free markov chain monte carlo method,” 2015.
"""
import time
import numpy as np
import copy

def expit(z):
    return np.exp(z) / (1 + np.exp(z))

def euclidean(z):
    return np.sum(z ** 2)

class BPS:
    def __init__(self, X, Y, prior_var, ref, store_skip):
        # X: input
        # Y: output
        # prior_var : variance of prior (normal distribution with mu = 0)
        # ref : refresh rate (1/ref)
        # store_skip : (thinned factor, not used)
        # you may adjust the sample counts by ""sample_time"" in function "sampler"
        
        self.X = X
        self.Y = Y
        self.all_samples = []
        self.prior_var = prior_var
        self.ref = ref
        self.store_skip = store_skip
        self.beta = np.random.normal(0,1,2)
        self.all_samples.append(self.beta)
        self.v = np.random.normal(0,1,2)
        self.count = 0
        self.pre_beta = self.beta
        self.success = X[np.where(Y == 1)]
        self.failure = X[np.where(Y == 0)]
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        # for storage time checking
        self.all_storage_time = 0
        self.after_burnin_storage_time = 0
        self.after_burnin_switch = 0
        
        
    
    def prior_local_upper_bound(self, t):
        # prior local upper bound calculation
        temp = np.max([0,np.dot((self.beta + self.v * t), self.v)]) * self.prior_var**-2
        return temp
    
    def constant_int(self):
        # the likelihood bound
        term1 = np.abs(self.v[0]) * len(self.success) * (self.v[0] < 0)
        term2 = np.max([0, np.abs(self.v[1]) * np.sum(self.success)])
        term3 = self.v[0] * len(self.failure) * (self.v[0] > 0)
        term4 = np.max([0, self.v[1] * np.sum(self.failure)])   
        return np.max([0, term1 + term2 + term3 + term4])
    
    def R_prior(self):
        # velocity bounce formula for prior
        nominator = np.dot(self.beta* self.prior_var**-2, self.v)
        denominator = euclidean(self.beta* self.prior_var**-2)
        new_v = self.v - 2*(nominator/denominator) * self.beta * self.prior_var**-2
        return new_v
    
    def g_constant(self):
        # exact likelihood intensity
        term_1 = np.sum(expit(self.beta[0] + self.beta[1]*self.success)) - len(self.success)
        term_2 = np.sum(self.success * (expit(self.beta[0] + self.beta[1]*self.success) -1))
        term_3 = np.sum(expit(self.beta[0] + self.beta[1]*self.failure))
        term_4 = np.sum(self.failure * (expit(self.beta[0] + self.beta[1]*self.failure)))
        gU = np.array([term_1 + term_3, term_2 + term_4])

        return np.dot(gU, self.v)
    
    def R_constant(self):
        # velocity bounce formula for likelihood
        term_1 = np.sum(expit(self.beta[0] + self.beta[1]*self.success)) - len(self.success)
        term_2 = np.sum(self.success * (expit(self.beta[0] + self.beta[1]*self.success) -1))
        term_3 = np.sum(expit(self.beta[0] + self.beta[1]*self.failure))
        term_4 = np.sum(self.failure * (expit(self.beta[0] + self.beta[1]*self.failure)))
        gU = np.array([term_1 + term_3, term_2 + term_4])
        nominator = np.dot(gU, self.v)
        denominator = euclidean(gU)
        new_v = self.v - 2*(nominator/denominator) * gU
        return new_v
    
    def storage(self, t_move, global_clock, iterations):
        # sample saved
        a = time.time()
        pre_beta = copy.deepcopy(self.pre_beta)
        pre_global_clock = global_clock - t_move
        self.pre_beta = copy.deepcopy(self.beta)
        # save the samples by sample_time (discretization)
        while(self.accumulate_tau <= global_clock):
            now = pre_beta + self.v * (self.accumulate_tau - pre_global_clock)
            self.accumulate_tau += self.sample_time
            self.all_samples.append(np.array(now))
            self.sample_count += 1
        b = time.time()
        self.all_storage_time += b-a
        if self.after_burnin_switch:
            self.after_burnin_storage_time += b-a 
        if self.count % self.verbose == 0 and self.count != 0:
            print('Current loop exec: ' + str(self.count))
            print('Current process: ' + str(self.sample_count))
            print('Now global clock: ' + str(global_clock))
        if(self.count >= iterations-1):
            print('Current loop exec: ' + str(self.count))
            print('Current process: ' + str(self.sample_count))
            print('Now global clock: ' + str(global_clock))
            print('====bounce counts===')
            print('Prior bounce: '+ str(self.prior_bounce))
            print('Posterior bounce :' + str(self.post_bounce))
            print('Refresh count: ' + str(self.ref_count))
            self.switch = 0
    
    def sampler(self, delta, sample_time, iterations, burninIters, verbose):
        # delta: valid time upperbound
        # sample_time : discretization for continuous path
        # iterations : iters to loop
        # burninIters : iterations to end the burn-in process
        # verbose : log the current progress by "verbose"
        
        global_clock = 0
        temp_clock = delta
        self.prior_bounce = 0
        self.post_bounce = 0
        self.ref_count = 0
        self.sample_time = sample_time
        self.accumulate_tau = sample_time
        self.sample_count = 0
        prior_bound = self.prior_local_upper_bound(delta)
        self.verbose = verbose

        self.switch = 1
        while(self.switch):
            self.count += 1
            constant_bound = self.constant_int()
            total_bound = prior_bound + constant_bound + self.ref
            tau = np.random.exponential(1/total_bound, size = 1)
            if (global_clock + tau) > temp_clock:
                self.beta = self.beta + self.v * (temp_clock - global_clock)
                prior_bound = self.prior_local_upper_bound(delta)
                global_clock = temp_clock
                self.storage(temp_clock - global_clock, global_clock, iterations)
                temp_clock = temp_clock + delta
            else:
                self.beta = self.beta + self.v * tau
                self.storage(tau, global_clock+tau, iterations)
                j = np.random.choice(3, 1, replace=False, p=np.array([prior_bound, constant_bound, self.ref]) / total_bound)
                u = np.random.uniform(0,1,1)
                if j == 0:
                    nominator = self.prior_var**-2 * np.max([0,np.dot(self.beta, self.v)])
                    if u < (nominator / prior_bound):
                        self.v = self.R_prior()
                        self.prior_bounce += 1
                elif j == 1:
                    nominator = np.max([0, self.g_constant()])
                    if u < (nominator / constant_bound):
                        self.v = self.R_constant()
                        self.post_bounce += 1
                elif j == 2:
                    self.v = np.random.normal(0,1,2)
                    self.ref_count += 1
                
                prior_bound = self.prior_local_upper_bound(temp_clock - global_clock - tau)
                global_clock = global_clock + tau
            
            if self.count == burninIters:
                self.burnin_time = time.time()
                self.burnin_sample = copy.deepcopy(self.sample_count)
                self.after_burnin_switch = 1
                
                
                
                
                
                
                
                
                