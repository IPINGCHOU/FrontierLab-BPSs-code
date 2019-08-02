# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:57:47 2019

@Title: FrontierLab exchange program - DBPS source code (for Neural Networks)
@Author: Chou I-Ping
@Reference: C. Sherlock and A. H. Thiery, “A discrete bouncy particle sampler,” 2017.

The structre all almost the same as the code for Bayesian Logistic Regression
Just simply change the input, and apply tensorflow package for GPU acceleration on gradient calculation.
Check the comments in DBPS_BLR_util.py for more details.
"""

import numpy as np
import copy 
import tensorflow as tf
import time
import os
import logging
import gc
from imp import reload
reload(logging)

def ess(z):
    return np.sum(z)**2/np.sum(z**2)

def expit(z):
    return np.exp(z) / (1 + np.exp(z))

class DBPS:
    def __init__(self, X, Y, delta, store_skip, save_iter):
        # init input
        self.dataX = X
        self.dataY = Y[:,1].reshape(len(self.dataX), 1)
        self.X = tf.transpose(tf.convert_to_tensor(self.dataX, dtype = tf.float64))
        self.Y = tf.transpose(tf.convert_to_tensor(self.dataY, dtype = tf.float64))
        self.delta = delta
        self.store_skip = store_skip
        self.save_iter = save_iter
        self.save_part_count = 1
        # storage
        self.all_theta = []
        self.all_bias = []
        self.exec_time = 0
        # status
        self.count = 0
        self.stage1_prob = 0
        self.stage2_prob = 0
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        self.sample_count = 0
        
        # init parameters
        # change the input_size if the dataset changes
        self.input_size = 28*28
        self.hidden_size_1 = 25
        self.output_size = 1
        
#        theta1_temp = (np.random.normal(size = self.hidden_size_1 * self.input_size))
        theta1_temp = np.zeros(shape = self.hidden_size_1 * self.input_size)
        theta1      = np.reshape(theta1_temp[:self.hidden_size_1 * self.input_size],(self.hidden_size_1, self.input_size))
        theta1 = tf.convert_to_tensor(theta1, dtype = tf.float64)
#        theta2_temp = (np.random.normal(size = self.output_size * self.hidden_size_1))
        theta2_temp = np.zeros(shape = self.output_size * self.hidden_size_1)
        theta2      = np.reshape(theta2_temp[:self.output_size * self.hidden_size_1],(self.output_size, self.hidden_size_1))
        theta2 = tf.convert_to_tensor(theta2, dtype = tf.float64)
#        bias_1_temp = np.random.normal(size = self.hidden_size_1)
        bias_1_temp = np.zeros(shape = self.hidden_size_1)
        bias_1      = bias_1_temp.reshape(len(bias_1_temp),1)
        bias_1 = tf.convert_to_tensor(bias_1, dtype = tf.float64)
#        bias_2_temp = np.random.normal(size = self.output_size)
        bias_2_temp = np.zeros(shape = self.output_size)
        bias_2      = bias_2_temp.reshape(len(bias_2_temp),1)
        bias_2 = tf.convert_to_tensor(bias_2, dtype = tf.float64)

        self.theta_arr = [theta1, theta2] # initial weight
        self.bias_arr = [bias_1, bias_2] # initial bias
        
        vtheta1_temp = (np.random.normal(size = self.hidden_size_1 * self.input_size))
        vtheta1      = np.reshape(vtheta1_temp[:self.hidden_size_1 * self.input_size],(self.hidden_size_1, self.input_size))
        vtheta1 = tf.convert_to_tensor(vtheta1, dtype = tf.float64)
        vtheta2_temp = (np.random.normal(size = self.output_size * self.hidden_size_1))
        vtheta2      = np.reshape(vtheta2_temp[:self.output_size * self.hidden_size_1],(self.output_size, self.hidden_size_1))
        vtheta2 = tf.convert_to_tensor(vtheta2, dtype = tf.float64)
        vbias_1_temp = np.random.normal(size = self.hidden_size_1)
        vbias_1      = vbias_1_temp.reshape(len(vbias_1_temp),1)
        vbias_1 = tf.convert_to_tensor(vbias_1, dtype = tf.float64)
        vbias_2_temp = np.random.normal(size = self.output_size)
        vbias_2      = vbias_2_temp.reshape(len(vbias_2_temp),1)
        vbias_2 = tf.convert_to_tensor(vbias_2, dtype = tf.float64)
        
        self.tv_arr = [vtheta1,vtheta2] # initial weight velocity
        self.bv_arr = [vbias_1,vbias_2]  # initial bias velocity
            
        # for storage time checking
        self.all_storage_time = 0
        self.after_burnin_storage_time = 0
        self.after_burnin_switch = 0
        
    def t_storage(self):
        if self.count % self.store_skip == 0:
            temp_theta = copy.deepcopy(self.theta_arr)
            temp_bias = copy.deepcopy(self.bias_arr)
            for i in range(2):
                temp_theta[i] = temp_theta[i].numpy()
                temp_bias[i] = temp_bias[i].numpy()
            self.all_theta.append(temp_theta)
            self.all_bias.append(temp_bias)
            self.sample_count += 1
            
    def part_saving(self):
        route = '/home/user/chou/Py_BPSs_NN_TF/DBPS_weights'
        if len(self.all_theta) >= self.save_iter or self.count == self.iterations:
            temp = copy.deepcopy(np.array(self.all_theta))
            temp2 = copy.deepcopy(np.array(self.all_bias))
            print_k = np.log10(self.kappa)*10
            print_d = np.log10(self.delta)
            np.save(os.path.join(route ,'theta0_'+str(int(print_k))+'_delta'+str(int(print_d))+'_part_'+str(self.save_part_count)), temp[:,0])
            np.save(os.path.join(route ,'theta1_'+str(int(print_k))+'_delta'+str(int(print_d))+'_part_'+str(self.save_part_count)), temp[:,1])
            np.save(os.path.join(route ,'bias0_'+str(int(print_k))+'_delta'+str(int(print_d))+'_part_'+str(self.save_part_count)), temp2[:,0])
            np.save(os.path.join(route ,'bias1_'+str(int(print_k))+'_delta'+str(int(print_d))+'_part_'+str(self.save_part_count)), temp2[:,1])
            self.all_theta.clear()
            self.all_bias.clear()
            logging.debug('Part: '+str(self.save_part_count)+' saved') 
            self.save_part_count += 1
            del temp, temp2
            gc.collect()
    
    
    def log_pi_cal(self, theta_arr, bias_arr):
        t1 = theta_arr[0]
        t2 = theta_arr[1]
        b1 = bias_arr[0]
        b2 = bias_arr[1]
        l1 = tf.matmul(t1, self.X) + b1
        l1 = tf.nn.relu(l1)
        pred_y = tf.matmul(t2, l1) + b2
        prior = 1/2*(tf.reduce_sum(t1**2) + tf.reduce_sum(t2**2) + tf.reduce_sum(b1**2) + tf.reduce_sum(b2**2))
        target = self.constant * (np.sum(np.log(np.exp(pred_y)/(1+np.exp(pred_y)))*self.Y) + np.sum(np.log(1/(1+np.exp(pred_y)))*(1-self.Y))) - prior
        return target
    
    def grad_cal(self, theta_arr, bias_arr):
        t1 = theta_arr[0]
        t2 = theta_arr[1]
        b1 = bias_arr[0]
        b2 = bias_arr[1]
        with tf.device('/gpu:0'):
            with tf.GradientTape(persistent = True) as t:
                t.watch(t1)
                t.watch(t2)
                t.watch(b1)
                t.watch(b2)
                l1 = tf.matmul(t1, self.X) + b1
                l1 = tf.nn.relu(l1)
                l2 = tf.matmul(t2, l1) + b2
                output = tf.nn.sigmoid_cross_entropy_with_logits(logits = l2, labels = self.Y)
                
        grad_theta = [-self.constant * t.gradient(output, t1) - t1, -self.constant * t.gradient(output, t2) - t2]
        grad_bias = [-self.constant * t.gradient(output, b1) - b1, -self.constant * t.gradient(output, b2) - b2]
        return grad_theta, grad_bias
        
    def bounce_v(self, theta_arr, bias_arr):
        grad_theta, grad_bias = self.grad_cal(theta_arr, bias_arr)
        nominator = 0
        denominator = 0
        new_tv, new_bv = [],[]
        for i in range(2):
            nominator += tf.reduce_sum(tf.multiply(grad_theta[i], self.tv_arr[i]))
            nominator += tf.reduce_sum(tf.multiply(grad_bias[i], self.bv_arr[i]))
            denominator += tf.reduce_sum(tf.multiply(grad_theta[i], grad_theta[i]))
            denominator += tf.reduce_sum(tf.multiply(grad_bias[i], grad_bias[i]))
            
        constant = 2*tf.divide(nominator,denominator)   
        for j in range(2):
            new_tv.append(-self.tv_arr[j] + constant * grad_theta[j])
            new_bv.append(-self.bv_arr[j] + constant * grad_bias[j])
        del grad_theta, grad_bias, nominator, denominator
        return new_tv, new_bv
    
    def sto_data(self, sto):
        N = self.dataX.shape[0]
        ind = np.random.randint(0, N, sto)
        self.X = tf.transpose(tf.convert_to_tensor(self.dataX[ind], dtype = tf.float64))
        self.Y = tf.transpose(tf.convert_to_tensor(self.dataY[ind], dtype = tf.float64))
    
    def kappa_cal(self, kappa):
        temp = tf.constant([], dtype = tf.float64)
        for i in range(2):
            temp = tf.concat([temp, tf.reshape(self.tv_arr[i],[-1]), tf.reshape(self.bv_arr[i],[-1])],0)
        zeta = tf.random.normal((tf.shape(temp)[0],),0,1,dtype = tf.float64)
        zeta -= tf.reduce_sum(tf.multiply(zeta, temp)) * temp / tf.pow(tf.norm(temp),2)
        zeta /= tf.norm(zeta)
        new_v = (temp + np.sqrt(kappa) * np.sqrt(self.delta)*zeta) / np.sqrt(1+kappa*self.delta)
        
        pre_pointer = 0
        pointer = 0
        for j in range(2):
            # resize theta
            theta_size = tf.shape(self.tv_arr[j])
            pointer += tf.reduce_prod(theta_size)
            self.tv_arr[j] = tf.reshape(new_v[pre_pointer: pointer], theta_size)
            pre_pointer = pointer
            # resize bias
            bias_size = tf.shape(self.bv_arr[j])
            pointer += tf.reduce_prod(bias_size)
            self.bv_arr[j] = tf.reshape(new_v[pre_pointer: pointer], bias_size)
            pre_pointer = pointer
        del zeta, new_v

    def DBPS_sampler(self, iters, burninIters, verbose, kappa = 0, subset = 0, sto = 0):
        filename = '/home/user/chou/Py_BPSs_NN_TF/DBPS_weights/log.txt'
        logging.basicConfig(level=logging.NOTSET,filemode='a',format='%(asctime)s - %(levelname)s : %(message)s',  filename=filename)
        self.kappa = kappa
        logging.debug('delta = '+str(self.delta)+', kappa = '+str(self.kappa)+' start')
        start = time.time()
        self.iterations = iters
        if sto != 0:
            self.constant = len(self.dataX) / sto
        else:
            self.constant = 1
            
        while(1):
            if sto != 0:
                self.sto_data(sto)
                
            self.count += 1
            # stage1
            c_theta_arr, c_bias_arr = copy.deepcopy(self.theta_arr), copy.deepcopy(self.bias_arr)
            # preparing stage 1 parameters
            s1_theta_arr, s1_bias_arr, s1_tv_arr, s1_bv_arr = [],[],[],[]
            for i in range(2):
                s1_theta_arr.append(copy.deepcopy(self.theta_arr[i]+self.delta * self.tv_arr[i]))
                s1_bias_arr.append(copy.deepcopy(self.bias_arr[i]+self.delta * self.bv_arr[i]))
                s1_tv_arr.append(copy.deepcopy(-self.tv_arr[i]))
                s1_bv_arr.append(copy.deepcopy(-self.bv_arr[i]))
            log_pi_1, log_pi_2 = self.log_pi_cal(c_theta_arr, c_bias_arr), self.log_pi_cal(s1_theta_arr, s1_bias_arr)
            stage1_p = np.minimum(1, np.exp(log_pi_2-log_pi_1))
            if np.random.uniform(0,1,1) < stage1_p:
                self.theta_arr, self.bias_arr, self.tv_arr, self.bv_arr = s1_theta_arr, s1_bias_arr, s1_tv_arr, s1_bv_arr
                self.stage1_prob += 1
            else:
                # stage 2 (basic bps)
                if subset == 0:
                    s2_tv_arr, s2_bv_arr = self.bounce_v(s1_theta_arr, s1_bias_arr)
                    # preparing stage 2 parameters
                    s2_theta_arr, s2_bias_arr = [],[]
                    for j in range(2):
                        s2_theta_arr.append(copy.deepcopy(self.theta_arr[j] + self.delta * (self.tv_arr[j] - s2_tv_arr[j])))
                        s2_bias_arr.append(copy.deepcopy(self.bias_arr[j] + self.delta * (self.bv_arr[j] - s2_bv_arr[j])))
                    log_pi_3 = self.log_pi_cal(s2_theta_arr, s2_bias_arr)
                    stage2_p = np.minimum(1, (1-np.minimum(1, np.exp(log_pi_2-log_pi_3)))/(1-np.minimum(1,np.exp(log_pi_2-log_pi_1))) * np.exp(log_pi_3-log_pi_1))
                    if np.random.uniform(0,1,1) < stage2_p:    
                        self.theta_arr, self.bias_arr, self.tv_arr, self.bv_arr = s2_theta_arr, s2_bias_arr, s2_tv_arr, s2_bv_arr
                        self.stage2_prob += 1
#                else:
#                    zeta = np.random.uniform(0,1,2)
#                    zeta = zeta / np.linalg.norm(zeta)
#                    pred_y, grad_x = self.grad_cal(c_beta)
#                    zeta = np.sign(np.dot(zeta, -grad_x))*zeta
#                    r = np.dot(zeta, grad_x) / np.dot(self.v, grad_x)
#                    a = (r**2-1)/(2*r-2*np.dot(zeta,self.v))
#                    b = r-a
#                    s2_v = np.linalg.norm(self.v)/b*zeta - a/b*self.v
#                    s2_beta = self.beta + self.delta * (self.v-s2_v)
#                    log_pi_3 = self.log_pi_cal(s2_beta)
##                    print([pi_1, pi_2, pi_3])
#                    stage2_p = np.minimum(1, (1-np.minimum(1, np.exp(log_pi_2-log_pi_3)))/(1-np.minimum(1,np.exp(log_pi_2-log_pi_1))) * np.exp(log_pi_3-log_pi_1))
#                    if np.random.normal(0,1,1) < stage2_p:
#                        self.beta, self.v = s2_beta, s2_v
#                        self.storage()
#                        self.stage2_prob.append(stage2_p)
                    
            # flip
            for m in range(2):
                self.tv_arr[m] = -self.tv_arr[m]
                self.bv_arr[m] = -self.bv_arr[m]
            
            # third step: using kappa
            if kappa > 0:
                self.kappa_cal(kappa)
            
            # save the weight
            self.t_storage()
            
            if self.count % verbose == 0:
                filename = '/home/user/chou/Py_BPSs_NN_TF/DBPS_weights/log.txt'
                logging.basicConfig(level=logging.NOTSET,filemode='a',format='%(asctime)s - %(levelname)s : %(message)s',  filename=filename)
                logging.debug('Finished: '+str(self.count)) 
                print('Current counts:' + str(self.count))
            # save by part, reduce ram using
            self.part_saving()
            
            if self.count == burninIters:
                self.burnin_time = time.time()
                self.burnin_sample = copy.deepcopy(self.sample_count)
            
            if self.count > iters:
                print('Current counts:' + str(self.count))
                logging.debug('delta = '+str(self.delta)+', kappa = '+str(self.kappa)+'finished')
                logging.debug('Total execution time: ' + str(time.time()-start))
                logging.debug('Total stage 1 accept count: ' + str(self.stage1_prob))
                logging.debug('Total stage 2 accept count: ' + str(self.stage2_prob))
                print(str(iters) + ' finished')
                break