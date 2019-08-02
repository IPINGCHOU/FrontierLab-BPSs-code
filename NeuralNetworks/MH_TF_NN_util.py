# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:13:25 2019

@Title: FrontierLab exchange program - Metropolis-Hastings source code (for Neural Networks)
@Author: Chou I-Ping

The structre all almost the same as the code for Bayesian Logistic Regression
Just simply change the input, and apply tensorflow package for GPU acceleration on gradient calculation.
Check the comments in MH_BLR_util.py for more details
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
def expit(z):
    return np.exp(z) / (1 + np.exp(z))

class MH:
    def __init__(self, X, Y, store_skip):
        # init input
        self.dataX = X
        self.dataY = Y[:,1].reshape(len(self.dataX), 1)
        self.X = tf.transpose(tf.convert_to_tensor(self.dataX, dtype = tf.float64))
        self.Y = tf.transpose(tf.convert_to_tensor(self.dataY, dtype = tf.float64))
        self.store_skip = store_skip
        # storage
        self.all_theta = []
        self.all_bias = []
        self.exec_time = 0
        # status
        self.count = 0
        self.accept_count = 0
        self.save_part_count = 1
        self.route = '/home/user/chou/Py_BPSs_NN_TF/MH_weights'
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        
        # init parameters
        self.input_size = 400
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
        
    def log_pi_cal(self, theta_arr, bias_arr):
        t1 = theta_arr[0]
        t2 = theta_arr[1]
        b1 = bias_arr[0]
        b2 = bias_arr[1]
        l1 = tf.matmul(t1, self.X) + b1
        l1 = tf.nn.relu(l1)
        pred_y = tf.matmul(t2, l1) + b2
        prior = 1/2*(tf.reduce_sum(t1**2) + tf.reduce_sum(t2**2) + tf.reduce_sum(b1**2) + tf.reduce_sum(b2**2))
        target = (np.sum(np.log(np.exp(pred_y)/(1+np.exp(pred_y)))*self.Y) + np.sum(np.log(1/(1+np.exp(pred_y)))*(1-self.Y))) - prior
        return target    
    
    def t_storage(self):
        if self.count % self.store_skip == 0:
            temp_theta = copy.deepcopy(self.theta_arr)
            temp_bias = copy.deepcopy(self.bias_arr)
            for i in range(2):
                temp_theta[i] = temp_theta[i].numpy()
                temp_bias[i] = temp_bias[i].numpy()
            self.all_theta.append(temp_theta)
            self.all_bias.append(temp_bias)
            
    def part_saving(self):
        route = self.route
        if len(self.all_theta) >= self.save_iter or self.count == int(self.iterations):
            temp = copy.deepcopy(np.array(self.all_theta))
            temp2 = copy.deepcopy(np.array(self.all_bias))
            np.save(os.path.join(route ,'MH_theta0_iters_'+ str(int(self.iterations)) +'_part_'+str(self.save_part_count)), temp[:,0])
            np.save(os.path.join(route ,'MH_theta1_iters_'+ str(int(self.iterations)) +'_part_'+str(self.save_part_count)), temp[:,1])
            np.save(os.path.join(route ,'MH_bias0_iters_'+ str(int(self.iterations)) +'_part_'+str(self.save_part_count)), temp2[:,0])
            np.save(os.path.join(route ,'MH_bias1_iters_'+ str(int(self.iterations)) +'_part_'+str(self.save_part_count)), temp2[:,1])
            self.all_theta.clear()
            self.all_bias.clear()
            print('Part: '+str(self.save_part_count)+' saved')
            logging.debug('Part: '+str(self.save_part_count)+' saved') 
            self.save_part_count += 1
            del temp, temp2
            gc.collect()
    
    def MH_sampler(self, can_sd, burninIters, iterations, verbose, save_iter):
        filename = self.route + '/log_MH.txt'
        logging.basicConfig(level=logging.NOTSET,filemode='a',format='%(asctime)s - %(levelname)s : %(message)s',  filename=filename)
        logging.debug('MH start') 
        self.save_iter = save_iter
        self.iterations = iterations
        self.t_storage()
        cur_lp = self.log_pi_cal(self.theta_arr, self.bias_arr)
        self.iterations = iterations
        
        for i in range(1, int(iterations+1),1):
            self.count += 1
            c_theta, c_bias = copy.deepcopy(self.theta_arr), copy.deepcopy(self.bias_arr)
            for j in range(2):
                c_theta[j] = c_theta[j] + np.random.normal(c_theta[j], can_sd)
                c_bias[j] = c_bias[j] + np.random.normal(c_bias[j], can_sd)
                can_lp = self.log_pi_cal(c_theta, c_bias)
                R = np.exp(can_lp - cur_lp)
                U = np.random.uniform(0,1,1)
                if U < R:
                    self.theta_arr = c_theta
                    self.bias_arr = c_bias
                    self.accept_count += 1
            
            self.t_storage()
            self.part_saving()
            
            if self.count % verbose == 0:
                print('Current process: ' + str(self.count))
                logging.debug('Finished: '+str(self.count)) 
            if i == burninIters:
                self.burnin_sample = copy.deepcopy(i)
                self.burnin_time = time.time()
