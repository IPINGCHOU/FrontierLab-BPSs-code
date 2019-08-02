# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:28:17 2019

@Title: FrontierLab exchange program - SBPS source code (for Neural Networks)
@Author: Chou I-Ping
@Reference1: A. Bouchard-Ct, S. J. Vollmer, and A. Doucet, “The bouncy particle sampler: A non-reversible rejection-free markov chain monte carlo method,” 2015.
@Reference2: A. Pakman, D. Gilboa, D. Carlson, and L. Paninski, “Stochastic bouncy particle sampler,” 2016.
@Reference3: https://github.com/dargilboa/SBPS-public

The structre all almost the same as the code for Bayesian Logistic Regression
Just simply change the input, and apply tensorflow package for GPU acceleration on gradient calculation.
Check the comments in BPS_BLR_util.py for more details.
"""
import tensorflow as tf
import numpy as np
import copy
import time
import logging
import os
import gc
from imp import reload
reload(logging)


class SBPS:
    def __init__(self,X,Y,T,dt,variance,mini_batch,save_iter):
        self.X = X
        self.Y = Y
        self.data = np.column_stack((X,Y))
        self.N = mini_batch
        self.M = len(X)
        self.T = T
        self.dt = dt
        self.variance = tf.convert_to_tensor(variance, dtype = tf.float64)
        self.cur_exec_time = time.time()
        self.start_time = time.time()
        self.save_iter = save_iter
        self.save_part_count = 1
        
        if self.N == len(self.X):
            self.route = '/home/user/chou/Py_BPSs_NN_TF/BPS_weights'
            self.filename = self.route + '/log_BPS.txt'
        else:
            self.route = '/home/user/chou/Py_BPSs_NN_TF/SBPS_weights'
            self.filename = self.route + '/log_SBPS.txt'
        
        # parameter initialize
        self.lh_theta = []
        self.lh_bias = []
        self.lh_bound = 0
        self.prior_theta_bound = 0
        self.prior_bias_bound = 0
        # stauts return
        self.count = 0
        self.theta_prior_bounce_count =  0
        self.bias_prior_bounce_count = 0
        self.likelihood_bounce_count = 0
        self.ref_count = 0

        self.all_Ts = []
        self.all_proposal_time = []
        
        # burnin time is computer time
        # not BPS clock
        self.burnin_time = 0
        self.burnin_sample = 0
        self.sample_count = 0
        
        # fix the input_size if the dataset changes
        self.input_size = 28*28
        input_size = self.input_size
        hidden_size_1 = 25
        output_size = 1
#        theta1_temp = (np.random.normal(size = hidden_size_1 * input_size))
        theta1_temp = np.zeros(shape = hidden_size_1 * input_size)
        theta1      = np.reshape(theta1_temp[:hidden_size_1 * input_size],(hidden_size_1, input_size))
        theta1 = tf.convert_to_tensor(theta1, dtype = tf.float64)
#        theta2_temp = (np.random.normal(size = output_size * hidden_size_1))
        theta2_temp = np.zeros(shape = output_size * hidden_size_1)
        theta2      = np.reshape(theta2_temp[:output_size * hidden_size_1],(output_size, hidden_size_1))
        theta2 = tf.convert_to_tensor(theta2, dtype = tf.float64)
#        bias_1_temp = np.random.normal(size = hidden_size_1)
        bias_1_temp = np.zeros(shape = hidden_size_1)
        bias_1      = bias_1_temp.reshape(len(bias_1_temp),1)
        bias_1 = tf.convert_to_tensor(bias_1, dtype = tf.float64)
#        bias_2_temp = np.random.normal(size = output_size)
        bias_2_temp = np.zeros(shape = output_size)
        bias_2      = bias_2_temp.reshape(len(bias_2_temp),1)
        bias_2 = tf.convert_to_tensor(bias_2, dtype = tf.float64)

        self.theta_arr = np.array([theta1, theta2]) # initial weight
        self.bias_arr = np.array([bias_1, bias_2]) # initial bias
        self.pre_theta = copy.deepcopy(self.theta_arr)
        self.pre_bias = copy.deepcopy(self.bias_arr)
        
#         self.all_theta1 = np.array(theta1)
#         self.all_theta2 = np.array(theta2)
#         self.all_bias1 = np.array(bias_1)
#         self.all_bias2 = np.array(bias_2)
        self.all_theta1 = [np.array(theta1)]
        self.all_theta2 = [np.array(theta2)]
        self.all_bias1 = [np.array(bias_1)]
        self.all_bias2 = [np.array(bias_2)]
  
        vtheta1_temp = (np.random.normal(size = hidden_size_1 * input_size))
        vtheta1      = np.reshape(vtheta1_temp[:hidden_size_1 * input_size],(hidden_size_1, input_size))
        vtheta1 = tf.convert_to_tensor(vtheta1, dtype = tf.float64)
        vtheta2_temp = (np.random.normal(size = output_size * hidden_size_1))
        vtheta2      = np.reshape(vtheta2_temp[:output_size * hidden_size_1],(output_size, hidden_size_1))
        vtheta2 = tf.convert_to_tensor(vtheta2, dtype = tf.float64)
        vbias_1_temp = np.random.normal(size = hidden_size_1)
        vbias_1      = vbias_1_temp.reshape(len(vbias_1_temp),1)
        vbias_1 = tf.convert_to_tensor(vbias_1, dtype = tf.float64)
        vbias_2_temp = np.random.normal(size = output_size)
        vbias_2      = vbias_2_temp.reshape(len(vbias_2_temp),1)
        vbias_2 = tf.convert_to_tensor(vbias_2, dtype = tf.float64)
        
        self.tv_arr = [vtheta1,vtheta2] # initial weight velocity
        self.bv_arr = [vbias_1,vbias_2]  # initial bias velocity
        
        # for storage time checking
        self.all_storage_time = 0
        self.after_burnin_storage_time = 0
        self.after_burnin_switch = 0
        
    def storage(self, t_move, g_clock):
        if self.LineIntegral:
            a = time.time()
            while(self.accumulate_tau <= g_clock):
                pre_global_clock = g_clock - t_move
#                 now_theta = self.pre_theta + np.array(self.tv_arr) * (self.accumulate_tau - pre_global_clock)
#                 now_bias = self.pre_bias + np.array(self.bv_arr) * (self.accumulate_tau - pre_global_clock)
                now_theta, now_bias = [],[]
                for i in range(2):
                    now_theta.append(self.pre_theta[i] + self.tv_arr[i] * (self.accumulate_tau - pre_global_clock))
                    now_bias.append(self.pre_bias[i] + self.bv_arr[i] * (self.accumulate_tau - pre_global_clock))
                self.accumulate_tau += self.sample_time
                self.sample_count += 1
                now_theta = copy.deepcopy(now_theta)
                now_bias = copy.deepcopy(now_bias)
#                 self.all_theta1.append(copy.deepcopy(now_theta[0].numpy()))
#                 self.all_theta2.append(copy.deepcopy(now_theta[1].numpy()))
#                 self.all_bias1.append(copy.deepcopy(now_bias[0].numpy()))
#                 self.all_bias2.append(copy.deepcopy(now_bias[1].numpy()))
                self.all_theta1.append(now_theta[0].numpy())
                self.all_theta2.append(now_theta[1].numpy())
                self.all_bias1.append(now_bias[0].numpy())
                self.all_bias2.append(now_bias[1].numpy())
                self.all_Ts.append(g_clock)
                del now_theta, now_bias
            self.pre_theta = copy.deepcopy(self.theta_arr)
            self.pre_bias = copy.deepcopy(self.bias_arr)
            b = time.time()
            self.all_storage_time += b - a
            if self.after_burnin_switch:
                self.after_burnin_storage_time += b - a
        else:
            a = time.time()
#             self.all_theta.append(copy.deepcopy(self.theta_arr))
#             self.all_bias.append(copy.deepcopy(self.bias_arr))
            self.all_theta1.append(now_theta[0].numpy())
            self.all_theta2.append(now_theta[1].numpy())
            self.all_bias1.append(now_bias[0].numpy())
            self.all_bias2.append(now_bias[1].numpy()) 
            b = time.time()
            self.all_storage_time += b-a
            if self.after_burnin_switch:
                self.after_burnin_storage_time += b-a

    def part_saving(self):
        route = self.route
        if len(self.all_theta1) >= self.save_iter or self.count == self.iteration:
            np.save(os.path.join(route ,'SBPS_theta0_mini-batch_'+str(self.N)+'_ref_'+str(self.ref)+'_part_'+str(self.save_part_count)), np.array(self.all_theta1))
            np.save(os.path.join(route ,'SBPS_theta1_mini-batch_'+str(self.N)+'_ref_'+str(self.ref)+'_part_'+str(self.save_part_count)), np.array(self.all_theta2))
            np.save(os.path.join(route ,'SBPS_bias0_mini-batch_'+str(self.N)+'_ref_'+str(self.ref)+'_part_'+str(self.save_part_count)), np.array(self.all_bias1))
            np.save(os.path.join(route ,'SBPS_bias1_mini-batch_'+str(self.N)+'_ref_'+str(self.ref)+'_part_'+str(self.save_part_count)), np.array(self.all_bias2))
            self.all_theta1.clear()
            self.all_bias1.clear()
            self.all_theta2.clear()
            self.all_bias2.clear()
            print('Part: '+str(self.save_part_count)+' saved')
            logging.debug('Part: '+str(self.save_part_count)+' saved') 
            self.save_part_count += 1
    
    def sto_data(self):
        X_count = self.data.shape[0]
        if(self.N != X_count):
            col = self.data.shape[1]
            ind = np.random.randint(0, X_count, self.N)
            sto_data = self.data[ind, :]
            y = sto_data[:, col-1].reshape(self.N, 1)
            x = np.delete(sto_data, [col-2, col-1], axis = 1)
        else:
            x = self.X
            y = self.Y[:,1].reshape(len(self.X), 1)
            
        return tf.transpose(tf.convert_to_tensor(x, dtype = tf.float64)), tf.transpose(tf.convert_to_tensor(y, dtype = tf.float64))
    
    def likelihood_partial(self, x, y, bound):
        t1 = self.theta_arr[0]
        t2 = self.theta_arr[1]
        b1 = self.bias_arr[0]
        b2 = self.bias_arr[1]
        # split success and failure
        
        with tf.device('/gpu:0'):
            with tf.GradientTape(persistent = True) as t:
                with tf.GradientTape(persistent = True) as t_2:
                    t.watch(t1)
                    t.watch(t2)
                    t.watch(b1)
                    t.watch(b2)
                    t_2.watch(t1)
                    t_2.watch(t2)
                    t_2.watch(b1)
                    t_2.watch(b2)
                    l1 = tf.matmul(t1, x) + b1
                    l1 = tf.nn.relu(l1)
                    l2 = tf.matmul(t2, l1) + b2
                    with t_2.stop_recording():
                        output = tf.nn.sigmoid_cross_entropy_with_logits(logits = l2, labels = y)
                        
        constant = self.M / self.N
        self.lh_theta = [constant * t.gradient(output, t1), constant * t.gradient(output, t2)]
        self.lh_bias = [constant * t.gradient(output, b1), constant * t.gradient(output, b2)]
        
        if bound == True:
            self.lh_bound = 0
            self.lh_bound += constant * tf.nn.relu(tf.reduce_sum(tf.multiply(t_2.gradient(l2, t1), self.tv_arr[0])))
            self.lh_bound += constant * tf.nn.relu(tf.reduce_sum(tf.multiply(t_2.gradient(l2, t2), self.tv_arr[1])))
            self.lh_bound += constant * tf.nn.relu(tf.reduce_sum(tf.multiply(t_2.gradient(l2, b1), self.bv_arr[0])))
            self.lh_bound += constant * tf.nn.relu(tf.reduce_sum(tf.multiply(t_2.gradient(l2, b2), self.bv_arr[1])))
        del t, t_2, constant
            
    def prior_bound(self, t):
        self.prior_theta_bound, self.prior_bias_bound = 0,0
        for i in range(2):
            self.prior_theta_bound += tf.divide(tf.reduce_sum(tf.multiply(self.theta_arr[i] + self.tv_arr[i] * t, self.tv_arr[i])), self.variance**2)
            self.prior_bias_bound += tf.divide(tf.reduce_sum(tf.multiply(self.bias_arr[i] + self.bv_arr[i] * t, self.bv_arr[i])), self.variance**2)
        
        self.prior_bias_bound = tf.nn.relu(self.prior_bias_bound)
        self.prior_theta_bound = tf.nn.relu(self.prior_theta_bound)
            
    def likelihood_bounce(self):
        nominator = 0
        denominator = 0
        for i in range(2):
            nominator += tf.reduce_sum(tf.multiply(self.lh_theta[i], self.tv_arr[i]))
            nominator += tf.reduce_sum(tf.multiply(self.lh_bias[i], self.bv_arr[i]))
            denominator += tf.reduce_sum(tf.multiply(self.lh_theta[i], self.lh_theta[i]))
            denominator += tf.reduce_sum(tf.multiply(self.lh_bias[i], self.lh_bias[i]))
            
        constant = 2*tf.divide(nominator,denominator)   
        for j in range(2):
            self.tv_arr[j] = self.tv_arr[j] - constant * self.lh_theta[j]
            self.bv_arr[j] = self.bv_arr[j] - constant * self.lh_bias[j]
        del nominator, denominator
        
    def prior_bounce(self, g, v):
        nominator = 0
        denominator = 0
        for i in range(2):
            nominator += tf.reduce_sum(tf.multiply(g[i],v[i]))
            denominator += tf.reduce_sum(tf.multiply(g[i],g[i]))
        constant = 2*tf.divide(nominator, denominator)
        for j in range(2):
            v[j] = v[j] - constant * g[j]     
        del nominator, denominator , constant
            
        return v
            
    def ref_v(self):
        input_size = self.input_size
        hidden_size_1 = 25
        output_size = 1
        
        vtheta1_temp = (np.random.normal(size = hidden_size_1 * input_size))
        vtheta1      = np.reshape(vtheta1_temp[:hidden_size_1 * input_size],(hidden_size_1, input_size))
        vtheta1 = tf.convert_to_tensor(vtheta1, dtype = tf.float64)
        vtheta2_temp = (np.random.normal(size = output_size * hidden_size_1))
        vtheta2      = np.reshape(vtheta2_temp[:output_size * hidden_size_1],(output_size, hidden_size_1))
        vtheta2 = tf.convert_to_tensor(vtheta2, dtype = tf.float64)
        vbias_1_temp = np.random.normal(size = hidden_size_1)
        vbias_1      = vbias_1_temp.reshape(len(vbias_1_temp),1)
        vbias_1 = tf.convert_to_tensor(vbias_1, dtype = tf.float64)
        vbias_2_temp = np.random.normal(size = output_size)
        vbias_2      = vbias_2_temp.reshape(len(vbias_2_temp),1)
        vbias_2 = tf.convert_to_tensor(vbias_2, dtype = tf.float64)
        
        self.tv_arr = [vtheta1,vtheta2] # initial weight velocity
        self.bv_arr = [vbias_1,vbias_2]  # initial bias velocity
        del vtheta1, vtheta2, vbias_1, vbias_2
        
    def arr_move(self, arr, v, t):
        for i in range(2):
            arr[i] += v[i]*t
        return arr
        
    def SBPS_sampler(self, ref, sample_time, iteration, burninIters, verbose, LineIntegral = 1):
        logging.basicConfig(level=logging.NOTSET,filemode='a',format='%(asctime)s - %(levelname)s : %(message)s',  filename=self.filename)
        logging.debug('SBPS start')
        start = time.time()
        # initialize parameter
        g_clock = 0
        t_clock = self.dt
        self.prior_bound(self.dt)
        # initialize storing list
        self.sample_time = sample_time
        self.accumulate_tau = sample_time
        self.iteration = iteration
        self.LineIntegral = LineIntegral
        self.ref = ref
        
        while(1):
            x,y = self.sto_data()
            self.likelihood_partial(x,y, True)
            total_bound = self.lh_bound + self.prior_theta_bound + self.prior_bias_bound + ref
            tau = np.random.exponential(1/total_bound)
            
            if(g_clock + tau > t_clock):
                self.theta_arr = self.arr_move(self.theta_arr, self.tv_arr, t_clock-g_clock)
                self.bias_arr = self.arr_move(self.bias_arr, self.bv_arr, t_clock-g_clock)
                self.storage(t_clock - g_clock, g_clock)
                self.prior_bound(self.dt)
                g_clock = t_clock
                t_clock += self.dt
                self.count += 1
            else:
                self.theta_arr = self.arr_move(self.theta_arr, self.tv_arr, tau)
                self.bias_arr = self.arr_move(self.bias_arr, self.bv_arr, tau)    
                self.storage(tau, g_clock+tau)
                j = np.random.choice(4, 1, p = np.array([ref, self.prior_theta_bound, self.prior_bias_bound, self.lh_bound]) / total_bound)
                u = np.random.uniform(0,1,1)
                
                if j==0:
                    self.ref_v()
                    self.ref_count += 1
                elif j==1:
                    temp_prior = 0
                    for i in range(2):
                        temp_prior += tf.reduce_sum(tf.multiply(self.theta_arr[i], self.tv_arr[i]))
                    temp_prior = tf.divide(tf.nn.relu(temp_prior), self.variance**2)
                    if u < tf.divide(temp_prior, self.prior_theta_bound):
                        self.tv_arr = self.prior_bounce(self.theta_arr, self.tv_arr)
                        self.theta_prior_bounce_count += 1
                elif j==2:
                    temp_prior = 0
                    for i in range(2):
                        temp_prior += tf.reduce_sum(tf.multiply(self.bias_arr[i], self.bv_arr[i]))
                    temp_prior = tf.divide(tf.nn.relu(temp_prior), self.variance**2)
                    if u < tf.divide(temp_prior, self.prior_bias_bound):
                        self.bv_arr = self.prior_bounce(self.bias_arr, self.bv_arr)
                        self.bias_prior_bounce_count += 1
                elif j==3:
                    self.likelihood_partial(x,y,False)
                    temp_lh = 0
                    for i in range(2):
                        temp_lh += tf.reduce_sum(tf.multiply(self.lh_theta[i], self.tv_arr[i]))
                        temp_lh += tf.reduce_sum(tf.multiply(self.lh_bias[i], self.bv_arr[i]))
                    if u < tf.divide(tf.nn.relu(temp_lh), self.lh_bound):
                        self.likelihood_bounce()
                        self.likelihood_bounce_count += 1
                
                self.prior_bound(t_clock - g_clock - tau)
                g_clock += tau
                self.count += 1

            if(self.count % verbose == 0):
                logging.debug('Iterations: '+str(self.count) + ' Samples: ' + str(self.sample_count) + ' Current clock:' + str(g_clock)) 
                print('Current clock: ' + str(g_clock))
                print('Current counts: ' + str(self.count))
            self.part_saving()
            
            if self.count == burninIters:
                self.burnin_time = time.time()
                self.burnin_sample = copy.deepcopy(self.sample_count)
                self.after_burnin_switch = 1
            
            if(self.count > iteration):
                print('Current clock: ' + str(g_clock))
                print('Current counts: ' + str(self.count))
                self.exec_time = str(time.time()-start)
                logging.debug('Finished: '+str(self.count))
                logging.debug('Total exec time' + self.exec_time)
                logging.debug('theta prior bounce counts' + str(self.theta_prior_bounce_count))
                logging.debug('bias prior bounce counts' + str(self.bias_prior_bounce_count))
                logging.debug('LH bounce counts' + str(self.likelihood_bounce_count))
                logging.debug('Refresh counts' + str(self.ref_count))
                break
