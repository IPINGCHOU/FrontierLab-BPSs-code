# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:57:30 2019

@Title: FrontierLab exchange program - SBPS source code (for Bayesian Logistic Regression)
@Author: Chou I-Ping
@Reference1: A. Pakman, D. Gilboa, D. Carlson, and L. Paninski, “Stochastic bouncy particle sampler,” 2016.
@Reference2: https://github.com/dargilboa/SBPS-public
"""

import time
import numpy as np
import copy

def expit(z):
#    print(z)
    return np.exp(z) / (1 + np.exp(z))

def euclidean(z):
    return np.sum(z ** 2)

def ess(z):
    return np.sum(z)**2/np.sum(z**2)

#%%

class SBPS:
    def __init__(self, X, Y, T, dt, up_factor, prior_mean, priorG_var, mini_batch, verbose, iterations, burninIters):
        # X: input
        # Y: output
        # T: accumulate time to stop the process (stop the loop by T or by iterations)
        # dt: time limit of valid local-in-time upperbound
        # up_factor: confidence band multiple for Local Bayesian Regression, check reference chapter 4 for details
        # prior_mean : prior mean for Local Bayesian Regression
        # priorG_var : prior var for local Bayesian Regression
        # mini-batch : batch size
        # verbose : log the current progress by "verbose"
        # iterations: iterations counts to run
        # burninIters : iterations to end the burn-in process
        self.X = X
        self.Y = Y
        self.data = np.array([X,Y])
        self.T = T
        self.dt = dt
        self.up_factor = up_factor
        self.prior_mean = prior_mean
        self.priorG_var = priorG_var
        self.N = mini_batch
        self.M = len(self.X)
        self.verbose = verbose
        self.iterations = iterations
        self.burninIters = burninIters
        
        self.t = 0
        self.max_rt = 10
        self.con = self.M / self.N
        self.this_time = 0
        self.global_time = 0
        self.gW = 0 # current gradient of tilde W
        self.G = 0 # current tilde G
        self.std = 0 # current c_ti^2
        self.var = 0 # current var[v * log p]
        self.intercept = 0
        self.beta1 = 1
        self.Sigma = np.zeros((2,2))
        self.Gs = []
        self.Ts = []
        self.Stds = []
        self.weights = []
        self.all_intercept = [self.intercept]
        self.all_beta1 = [self.beta1]
        # preconditioner settings
        self.grad2 = np.zeros(2)
        self.pre_beta1 = 0.99
        self.lam = 1e-4
        self.preconditioner = 1
        # status counts
        self.count = 0
        self.bounce_counts = 0
        self.accept_over_1 = 0
        self.bounce_prob = []
        self.all_Gs = []
        self.all_upper_bound = []
        self.all_Ts = []
        self.all_full_Gs = []
        self.all_bounce_Ts = []
        self.all_predGs = []
        self.err_bound = 0
        # initial particle pos and v
        self.samples = np.random.normal(0,1,2)
        self.velocity = np.random.normal(0,1,2)
        self.alt_samples = [copy.deepcopy(self.samples)]
        
        self.full_success, self.full_failure = self.sto_data(full = True)
        
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        # for storage time checking
        self.all_storage_time = 0
        self.after_burnin_storage_time = 0
        self.after_burnin_switch = 0
    
    def storage(self, t_move, global_clock, cur_samples):
        # sample storage
        a = time.time()
        pre_beta = copy.deepcopy(self.samples)
        pre_global_clock = global_clock - t_move
        self.samples = copy.deepcopy(cur_samples)
        # save the samples by sample_interval (discretization)
        while(self.accumulate_tau <= global_clock):
            now = pre_beta + self.velocity * (self.accumulate_tau - pre_global_clock)
            self.accumulate_tau += self.sample_interval
            self.alt_samples.append(np.array(now))
            self.sample_count += 1
        b = time.time()
        self.all_storage_time += b-a
        if self.after_burnin_switch:
            self.after_burnin_storage_time += b-a
        if self.count % self.verbose == 0 and self.count != 0:
            print('Current turn points:' + str(self.count))
            print('Current process: ' + str(self.sample_count))
            print('Now global clock: ' + str(global_clock))
        if(self.count >= self.iterations-1):
            print('Current turn points:' + str(self.count))
            print('Current process: ' + str(self.sample_count))
            print('Now global clock: ' + str(global_clock))
            self.switch = 0
            
#         if self.count % self.store_skip == 0:
#             self.all_beta.append(copy.deepcopy(self.beta))

    def generate_samples(self, sample_interval, store_skip = 1, use_preconditioner = False):
        # sample_interval : discretization
        # store_skip : thinned factor (not used)
        # use_preconditioner : preconditioner used or not, check chapter 5
        self.sample_interval = sample_interval
        self.accumulate_tau = sample_interval
        self.sample_count = 0
        self.switch = 1
        cur_samples = self.samples
        
        if use_preconditioner == False:
            while self.switch:
                success, failure = self.sto_data()
                self.tildeW_var(success, failure, use_preconditioner)
                self.tildeG(self.t, success, failure, use_preconditioner)
#                 self.full_G(self.t, self.full_success, self.full_failure)
                self.std_cal()
                self.this_time += self.t
                self.Ts.append(copy.copy(self.this_time))
                self.global_time += self.t
                self.all_Ts.append(self.global_time)
                
                bounce_t, upper_bound = self.sample_next()
                # check if full gradient exceeds upper bound
#                 if self.all_full_Gs[-1] > upper_bound:
#                     self.err_bound += 1
                    
                cur_samples = cur_samples + np.multiply(self.velocity, bounce_t)
                self.t = bounce_t
                              
                self.storage(bounce_t, self.global_time, cur_samples)
                
                # accept reject
                self.accept_reject(upper_bound, use_preconditioner)
                self.count += 1
        
                if self.count == self.burninIters:
                    self.burnin_time = time.time()
                    self.burnin_sample = copy.deepcopy(self.sample_count)
                    self.after_burnin_switch = 1
            
        else:
            while self.switch:
                success, failure = self.sto_data()
                self.cal_preconditioner()
                self.tildeW_var(success, failure, use_preconditioner)
                self.tildeG(self.t, success, failure, use_preconditioner)
#                 self.full_G(self.t, self.full_success, self.full_failure)
                self.std_cal()
                self.this_time += self.t
                self.Ts.append(copy.copy(self.this_time))
                self.global_time += self.t
                self.all_Ts.append(self.global_time)
                
                bounce_t, upper_bound = self.sample_next()
                # check if full gradient exceeds upper bound
#                 if self.all_full_Gs[-1] > upper_bound:
#                     self.err_bound += 1
                    
                cur_samples = cur_samples + np.multiply(self.preconditioner, np.multiply(self.velocity, bounce_t))
                self.t = bounce_t
                
                self.storage(bounce_t, self.global_time, cur_samples)
                
                # accept reject
                self.accept_reject(upper_bound, use_preconditioner)
                self.count += 1
            
#                if self.counts % self.verbose == 0:
#                    print('The current time: ' + str(self.global_time))
#                    print('Current total samples: ' + str(self.counts))
            
    def sto_data(self, full = False):
        # randomly select input data by mini-batch
        if full == False:
            ind = np.random.randint(0, self.data.shape[1], self.N)
        else:
            ind = np.random.randint(0, self.data.shape[1], self.M)
        sto_data = self.data[:, ind]
        success = sto_data[0][np.where(sto_data[1] == 1)]
        failure = sto_data[0][np.where(sto_data[1] == 0)]
        return np.array(success), np.array(failure)
        
    def tildeW_var(self, success, failure, use_preconditioner):
        # calculate tilde W and varanice for G estimation
        # tilde W part
        cur_s = self.samples
        s_term = cur_s[0] + success * cur_s[1]
        f_term = cur_s[0] + failure * cur_s[1]
        term1 = np.sum(expit(s_term)) - len(success)
        term2 = np.sum(success * (expit(s_term)-1))
        term3 = np.sum(expit(f_term))
        term4 = np.sum(failure * (expit(f_term)))
        # var part
        var_term = []
        term5 = np.array((expit(s_term)-1) * self.velocity[0]) + np.array((success * (expit(s_term)-1) * self.velocity[1]))
        term6 = np.array(expit(f_term) * self.velocity[0]) + np.array((failure * expit(f_term) * self.velocity[1]))
        
        if use_preconditioner == False:
            gW_term = [cur_s[0] + self.con*(term1+term3), cur_s[1] + self.con*(term2+term4)]
            var_term = np.concatenate([term5,term6])
            self.gW = np.array(gW_term)
            self.var = np.var(var_term)
        else:
            gW_term = [self.preconditioner[0] * (cur_s[0] + self.con*(term1+term3)), self.preconditioner[1] * (cur_s[1] + self.con*(term2+term4))]
            var_term = np.concatenate([self.preconditioner[0]*term5, self.preconditioner[1]*term6])
            self.gW = np.array(gW_term)
            self.var = np.var(var_term)
#        self.var = np.var(sum(var_term,[]))
        self.var = np.var(var_term)

    def cal_preconditioner(self):
        # preconditioner calculation
        self.grad2 = self.gW*self.gW*(1-self.pre_beta1) + self.pre_beta1*self.grad2
        self.preconditioner = 1 / (np.sqrt(self.grad2)+self.lam)
        self.preconditioner = self.preconditioner / float(np.mean(self.preconditioner))

    def tildeG(self, t, success, failure, use_preconditioner):
        # G calculation
        cur_s = self.samples
        
        if use_preconditioner == False:
            [inter, coef] = cur_s + self.velocity * t
        else:
            [inter, coef] = cur_s + self.preconditioner * self.velocity * t
            
        s_term = inter + success * coef
        f_term = inter + failure * coef
        term1 = np.sum(expit(s_term)) - len(success)
        term2 = np.sum(success * (expit(s_term)-1))
        term3 = np.sum(expit(f_term))
        term4 = np.sum(failure * (expit(f_term)))

        term5 = np.array([inter + self.con*(term1+term3), coef + self.con*(term2+term4)])
        if use_preconditioner == False:
            term5 = np.array([inter + self.con*(term1+term3), coef + self.con*(term2+term4)])
            self.all_Gs.append(np.dot(self.velocity, term5))
            self.G = np.maximum(0, np.dot(self.velocity, term5))
        else:
            term5 = np.array([self.preconditioner[0]*(inter + self.con*(term1+term3)), self.preconditioner[1]*(coef + self.con*(term2+term4))])
            self.all_Gs.append(np.dot(self.velocity, term5))
            self.G = np.maximum(0, np.dot(self.velocity, term5))
        self.Gs.append(self.G)
    
    def full_G(self, t, success, failure):
        # exact G calculation (for checking difference between exact G and estimated G, comment out in the code)
        cur_s = self.samples
        inter = cur_s[0] + self.velocity[0] * t
        coef = cur_s[1] + self.velocity[1] * t
        s_term = inter + success * coef
        f_term = inter + failure * coef
        term1 = np.sum(expit(s_term)) - len(success)
        term2 = np.sum(success * (expit(s_term)-1))
        term3 = np.sum(expit(f_term))
        term4 = np.sum(failure * (expit(f_term)))

        term5 = np.array([inter + self.con*(term1+term3), coef + self.con*(term2+term4)])
        self.all_full_Gs.append(np.dot(self.velocity, term5))
    
    def std_cal(self):
        # standard deviation calculate for Local Bayesian Regression
        self.std = (self.M**2/self.N) * (1-self.N/self.M) * self.var # c_ti^2
        self.Stds.append(self.std)
        self.weights.append(1/float(self.std))
    
    def G_cal(self, t):
        # estimated G
        return self.intercept + self.beta1 * t
    
    def pred_G(self, t):
        # predict G and calculate the upper bound of predicted G
        return self.G_cal(t), self.G_cal(t) + self.up_factor * np.sqrt(np.dot(np.dot([1,t],self.Sigma),[1,t]) + self.std)
    
    def sample_next(self):
        # propose G and upper bound
        u = -np.log(np.random.rand())
        predgs = []
        uppers = []
        rt = 0
        pg = 0
        prev_pg, prev_upper = self.pred_G(self.Ts[-1] + rt)
        predgs.append(prev_pg)
        uppers.append(prev_upper)
        
        while u > 0:
            
            if rt > self.max_rt:
                ri = prev_upper
                break
            
            
            rt += self.dt
            pg, upper = self.pred_G(self.Ts[-1] + rt)
            predgs.append(pg)
            uppers.append(upper)
            if upper > 0:
                cur_area = prev_upper * self.dt + (upper - prev_upper) * self.dt / 2
                if u > cur_area:
                    u -= cur_area
                else:
                    rt -= self.dt
                    a = (upper - prev_upper) / (2 * self.dt)
                    b = prev_upper
                    c = -u
                    lt = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                    rt += lt
                    
                    ri = prev_upper + lt*(upper - prev_upper)/self.dt
                    self.all_predGs.append(prev_pg + lt*(pg - prev_pg)/self.dt)
                    u = 0 # stop the loop
            prev_pg = pg
            prev_upper = upper
        
        return rt, ri
    
    def accept_reject(self, upper_bound, use_preconditioner):
        self.update_slope()
        
        self.all_upper_bound.append(upper_bound)
        if np.random.rand() < (np.maximum(0, self.G)/upper_bound):
            self.bounce_counts += 1
            self.bounce_prob.append(np.maximum(0, self.G)/upper_bound)
            self.bounce()
            if np.maximum(0, self.G)/upper_bound >= 1:
                self.accept_over_1 +=1
        else:
            self.all_bounce_Ts.append(float('nan'))
            self.clean()
    
    def bounce(self):
        self.velocity = self.velocity - 2 * (np.dot(self.velocity, self.gW)*self.gW/euclidean(self.gW))
        self.all_bounce_Ts.append(self.global_time)
#        print('bounced!')
    
    def clean(self):
        # clean after bounce event happens
        self.Gs = [-self.Gs[-1]]
        self.Stds = [self.Stds[-1]]
        self.Ts = [0]
        self.this_time = 0
        self.weights = [1/float(self.Stds[-1])]
        self.update_slope()
    
    def update_slope(self):
        # update slope for Local Bayesian Regression 
        vG = np.asarray(self.Gs)
        vT = np.asarray(self.Ts)
        vW = np.asarray(self.weights)
        Sigma0 = np.zeros((2,2))
        for i in range(len(self.Gs)):
            Sigma0 += self.weights[i] * np.outer([1, self.Ts[i]], [1, self.Ts[i]])
        Sigma0[1,1] += 1/float(self.priorG_var)
        self.Sigma = np.linalg.inv(Sigma0)
        
        [I_new,beta1_new] = np.dot(self.Sigma,[np.sum(vG*vW),np.sum(vG*vT*vW)+self.prior_mean/float(self.priorG_var)])
        self.beta1 = beta1_new
        self.intercept = I_new       
        self.all_beta1.append(self.beta1)
        self.all_intercept.append(self.intercept)

            
            