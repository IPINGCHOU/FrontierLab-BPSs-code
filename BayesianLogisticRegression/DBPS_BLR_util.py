# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:57:47 2019

@Title: FrontierLab exchange program - DBPS source code (for Bayesian Logistic Regression)
@Author: Chou I-Ping
@Reference: C. Sherlock and A. H. Thiery, “A discrete bouncy particle sampler,” 2017.
"""

import time
import numpy as np
import copy 

def ess(z):
    return np.sum(z)**2/np.sum(z**2)

def expit(z):
    return np.exp(z) / (1 + np.exp(z))

class DBPS_BLR:
    def __init__(self, X, Y, delta, store_skip):
        # X: input
        # Y: output
        # delta: time step size
        # store_skip: thinned factor
        
        
        # init input
        self.X = X
        self.Y = Y
        self.dataX = X
        self.dataY = Y
        self.delta = delta
        self.store_skip = store_skip
        # storage
        self.all_beta = []
        # status
        self.count = 0
        self.stage1_accept = 0
        self.stage2_accept = 0
        # init parameters
#        self.beta = np.random.normal(0,1,2)
        self.beta = np.float64([0,0])
        self.v = np.random.normal(0,1,2)
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        
    def storage(self):
        if self.count % self.store_skip == 0:
            self.all_beta.append(copy.deepcopy(self.beta))
    
    def log_pi_cal(self, beta):
        pred_y = np.multiply(beta[1], self.X) + beta[0]
        target = self.constant * (np.sum(np.log(np.exp(pred_y)/(1+np.exp(pred_y)))*self.Y) + np.sum(np.log(1/(1+np.exp(pred_y)))*(1-self.Y))) - 1/2*np.sum(beta**2)
        return target
    
    def grad_cal(self, beta):
        pred_y = np.multiply(beta[1], self.X) + beta[0]
        b1_temp = np.sum(self.Y * (np.exp(pred_y)*self.X/(1+np.exp(pred_y))-self.X)) + np.sum((1-self.Y) * np.exp(pred_y)*self.X/(1+np.exp(pred_y))) + beta[1]
        b0_temp = np.sum(self.Y * (np.exp(pred_y)/(1+np.exp(pred_y))-1)) + np.sum((1-self.Y) * np.exp(pred_y)/(1+np.exp(pred_y))) + beta[0]
        grad_x = np.array([-b0_temp, -b1_temp])
        return pred_y, grad_x
        
    def bounce_v(self, beta):
        pred_y, grad_x = self.grad_cal(beta)
        temp = -self.v + 2 * np.dot(self.v, grad_x) / np.sum(grad_x**2) * grad_x
        return temp
    
    def sto_data(self, sto):
        N = self.dataX.shape[0]
        ind = np.random.randint(0, N, sto)
        self.X = self.dataX[ind]
        self.Y = self.dataY[ind]
 
    
    def DBPS_sampler(self, burninIters, iterations, verbose, kappa = 0, subset = 0, sto = 0):
        if sto != 0:
            self.constant = len(self.dataX) / sto
        else:
            self.constant = 1
            
        while(1):
            if sto != 0:
                self.sto_data(sto)
                
            self.count += 1
            # stage1
            c_beta = copy.deepcopy(self.beta)
            s1_beta, s1_v = copy.deepcopy(self.beta + self.delta * self.v), copy.deepcopy(-self.v)
            log_pi_1, log_pi_2 = self.log_pi_cal(c_beta), self.log_pi_cal(s1_beta)
            stage1_p = np.minimum(1, np.exp(log_pi_2-log_pi_1))
            if np.random.uniform(0,1,1) < stage1_p:
                self.beta, self.v = s1_beta, s1_v
                self.stage1_accept += 1
            else:
                # stage 2 (basic bps)
                if subset == 0:
                    s2_v = self.bounce_v(s1_beta)
                    s2_beta = self.beta + self.delta*(self.v - s2_v)
                    log_pi_3 = self.log_pi_cal(s2_beta)
                    stage2_p = np.minimum(1, (1-np.minimum(1, np.exp(log_pi_2-log_pi_3)))/(1-np.minimum(1,np.exp(log_pi_2-log_pi_1))) * np.exp(log_pi_3-log_pi_1))
                    if np.random.uniform(0,1,1) < stage2_p:
                        self.beta, self.v = s2_beta, s2_v
                        self.stage2_accept += 1
                else:
                    zeta = np.random.uniform(0,1,2)
                    zeta = zeta / np.linalg.norm(zeta)
                    pred_y, grad_x = self.grad_cal(c_beta)
                    zeta = np.sign(np.dot(zeta, -grad_x))*zeta
                    r = np.dot(zeta, grad_x) / np.dot(self.v, grad_x)
                    a = (r**2-1)/(2*r-2*np.dot(zeta,self.v))
                    b = r-a
                    s2_v = np.linalg.norm(self.v)/b*zeta - a/b*self.v
                    s2_beta = self.beta + self.delta * (self.v-s2_v)
                    log_pi_3 = self.log_pi_cal(s2_beta)
#                    print([pi_1, pi_2, pi_3])
                    stage2_p = np.minimum(1, (1-np.minimum(1, np.exp(log_pi_2-log_pi_3)))/(1-np.minimum(1,np.exp(log_pi_2-log_pi_1))) * np.exp(log_pi_3-log_pi_1))
                    if np.random.normal(0,1,1) < stage2_p:
                        self.beta, self.v = s2_beta, s2_v
                        self.stage2_prob.append(stage2_p)
                    
            # flip
            self.v = -self.v
            
            # third step: using kappa
            if kappa > 0:               
                zeta = np.random.randn(2)
                zeta -= zeta.dot(self.v) * self.v / np.linalg.norm(self.v)**2
                zeta /= np.linalg.norm(zeta)
                    
                self.v = (self.v + np.sqrt(kappa)*np.sqrt(self.delta)*zeta) / np.sqrt(1+kappa*self.delta)
            del zeta
            
            # store weight
            self.storage()
            
            if self.count % verbose == 0:
                print('Current counts:' + str(self.count))
                
            if self.count == burninIters:
                self.burnin_time = time.time()
                self.burnin_sample = copy.deepcopy(self.count)
            
            if self.count > iterations:
                print('Current counts:' + str(self.count))
                print(str(iterations) + ' finished')
                break
                