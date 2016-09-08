# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:56:33 2016

@author: rob
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_rnn



class Model():
  def __init__(self,config):
    """Hyperparameters"""
    hidden_size = config['hidden_size']       	#hidden size of the LSTM
    batch_size = config['batch_size']    		# batch_size
    seq_len = config['seq_len']			# How long do you want the vectors with integers to add be?
    num_layers = config['num_layers']			# Number of RNN layers
    lr_rate = config['lr_rate']
    Nn = config['Nn']
    K = config['K']
    C = config['C']


    with tf.name_scope("Placeholders") as scope:
      #The place holders for the model
      self.inputs = tf.placeholder(tf.float32,shape=[batch_size,seq_len,Nn])
      self.target = tf.placeholder(tf.int32, shape=[batch_size,seq_len,Nn], name = 'Target')
      self.keep_prob = tf.placeholder("float", name = 'Drop_Out_keep_probability')
      #Processing of placeholders
      self.inputs_list = tf.unpack(tf.transpose(self.inputs,perm=[1,0,2]))
      self.target_hot = tf.one_hot(self.target,depth=C,dtype=tf.int32)  # now in [batch_size,seq_len,Nn,C]

    with tf.name_scope("Cell_fw") as scope:
      #Define one cell, stack the cell to obtain many layers of cell and wrap a DropOut
      cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size,use_peepholes=True)
#      cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers)
      cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,output_keep_prob=self.keep_prob)
      initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)

    with tf.name_scope("Cell_bw") as scope:
      #Define one cell, stack the cell to obtain many layers of cell and wrap a DropOut
      cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size)
#      cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers)
      cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,output_keep_prob=self.keep_prob)
      initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    with tf.name_scope("RNN") as scope:
      # Thanks to Tensorflow, the entire decoder is just one line of code:
      #outputs, states = seq2seq.rnn_decoder(inputs, initial_state, cell_fw)
      outputs, _, _ = bidirectional_rnn(cell_fw, cell_bw, self.inputs_list,
                          initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                          dtype=tf.float32)
      outputs_long = tf.concat(0, outputs)
    with tf.name_scope('FC') as scope:
      W_o = tf.get_variable('W_o',[2*hidden_size,Nn*K])
      b_o = tf.Variable(tf.constant(0.5, shape=[Nn*K]))
      V = tf.nn.xw_plus_b(outputs_long,W_o,b_o)
      V = tf.reshape(V,(seq_len,batch_size,Nn,K))
      V = tf.transpose(V,perm=[1,0,2,3])  #V now in [batch_size, seq_len, Nn, K]

    with tf.name_scope('Cost') as scope:
      #Both V and Y are now Tensors in [batch_size, seq_len, Nn, K]
      #This section implements equation (1) and (2) of http://arxiv.org/pdf/1508.04306.pdf
      self.Vn = tf.reshape(V,(batch_size,seq_len*Nn,K))
      Yn = tf.reshape(self.target_hot,(batch_size,seq_len*Nn,C))
      VVT = tf.batch_matmul(self.Vn,tf.transpose(self.Vn,[0,2,1]))  #V*V^T #in [batch_size, seq_len*Nn,seq_len*Nn]
      self.YYT = tf.batch_matmul(Yn,tf.transpose(Yn,[0,2,1]))  #Y*Y^T #in [batch_size, seq_len*Nn,seq_len*Nn]
      A = tf.sub(VVT,tf.cast(self.YYT,tf.float32))
      #Calculate (d,dw,W) for the Frobenius norm
      d = tf.reduce_sum(self.YYT,2,keep_dims=True)  #in [batch_size, seq_len*Nn, 1]
      dw = tf.div(1.0,tf.sqrt(tf.cast(d,tf.float32)))
      self.W = tf.batch_matmul(dw,tf.transpose(dw,[0,2,1]))  #in [batch_size, seq_len*Nn,seq_len*Nn]
      self.W = tf.stop_gradient(self.W)  #No need to backprop into W, these are constants
      norm = tf.reduce_sum(tf.square(tf.mul(self.W,A)),[1,2])
      self.cost = tf.reduce_mean(norm)

    with tf.name_scope("Optimization") as scope:
      global_step = tf.Variable(0, trainable=False)
      lr = tf.train.exponential_decay(lr_rate, global_step,
                                           1000, 0.1, staircase=False)
      optimizer = tf.train.AdamOptimizer(lr)
      self.train_op  = optimizer.minimize(self.cost,global_step=global_step)


    print('Finished computational graph')