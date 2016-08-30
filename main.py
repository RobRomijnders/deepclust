# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:33:35 2016

@author: rob
TODO
- Automate C from dataload D
"""

from dataload import DataLoad
import numpy as np
from model import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path



"""Hyperparameters"""
config = {}
config['Nn'] = Nn = 60
config['seq_len'] = seq_len = 30
config['hidden_size'] = 60       	#hidden size of the LSTM
config['batch_size'] = batch_size = 8    		# batch_size
config['num_layers'] = 1			# Number of RNN layers
config['lr_rate'] = 0.0005
config['C'] = C = 2
config['K'] = K = 10     #Embedding dimension

drop_out = 0.8 			# Drop out
max_iterations=200000		# Number of iterations to train with
plot_every = 20		# How often you want terminal output?


"""Load data"""
datadirec = '/home/rob/Dropbox/ml_projects/deepclust/data/saved/'

if os.path.isdir(datadirec):
  D = {}
  try:
    D['X_train'] = np.load(datadirec+'X_train.npy')
    D['y_train'] = np.load(datadirec+'y_train.npy')
    D['X_val'] = np.load(datadirec+'X_val.npy')
    D['y_val'] = np.load(datadirec+'y_val.npy')
    print('Read data from pickles')
  except:
    print('One or more files are missing from '+datadirec)
else:
  os.makedirs(datadirec)
  names = 3
  filenames = []
  for i in range(names):
    filenames.append(('/home/rob/Dropbox/ml_projects/deepclust/data/male'+str(i+1)+'.wav','/home/rob/Dropbox/ml_projects/deepclust/data/female'+str(i+1)+'.wav'))

  dl = DataLoad(seq_len,Nn,low_freq=6000)
  dl.read_data(filenames)
  dl.strip_zero()
  D = dl.return_data()

  for key, value in D.iteritems():
    np.save(datadirec+key+'.npy',value)
  print('Finished saving data')

#Obtain some sizes
N = D['X_train'].shape[0]
Nval = D['X_val'].shape[0]


model = Model(config)

#Fire up session
sess = tf.Session()
sess.run(tf.initialize_all_variables())


perf_collect = np.zeros((int(np.floor(max_iterations /plot_every)),7))
step = 0
print('Start backprop')
early_stop = False
ma_low = 10.0
k=0
while k < max_iterations and not early_stop:
  epoch = (k*batch_size)/N
  batch_ind = np.random.choice(N,batch_size,replace=False)
  X_batch = D['X_train'][batch_ind]
  y_batch = D['y_train'][batch_ind]


  #Create the dictionary of inputs to feed into sess.run
  train_dict = {model.inputs: X_batch, model.target: y_batch, model.keep_prob:drop_out}

  result = sess.run([model.train_op,model.cost],feed_dict=train_dict)   #perform an update on the parameters
  cost_train = result[1]
  if k == 0: ma_cost = cost_train
  ma_cost = 0.95*ma_cost + 0.05*cost_train


  if (k%plot_every==0):   #Output information
    batch_ind_sub = np.random.choice(Nval,batch_size,replace=False)
    X_val_sub = D['X_val'][batch_ind_sub]
    y_val_sub = D['y_val'][batch_ind_sub]

    val_dict = {model.inputs: X_val_sub, model.target: y_val_sub, model.keep_prob:1.0}
    result = sess.run([model.cost],feed_dict = val_dict )            #compute the cost on the validation set
    cost_val = result[0]

    perf_collect[step,0] = cost_train
    perf_collect[step,1] = cost_val

    #Code lines for EARLY STOPPING
    if step < 10:
      ma_cost = np.mean(perf_collect[:step,1])
    else:
      ma_cost = np.mean(perf_collect[step-10:step,1])
    if ma_cost < ma_low:
      ma_low = ma_cost
    if step > 15 and ma_cost > ma_low*1.15:
      early_stop = True
    perf_collect[step,2] = ma_cost
    print('At %4.0f/%6.0f - Cost Train:%5.3f(%5.3f) Val:%5.3f'%(k,max_iterations,cost_train,ma_cost,cost_val))
    step += 1
  k += 1


plt.figure()
plt.plot(perf_collect[:,0],label='train cost')
plt.plot(perf_collect[:,1],label='val cost')
plt.plot(perf_collect[:,6],label='val cost (ma)')
plt.legend()
plt.title('Prediction task')
plt.show()

"""Visualize the embedding space"""
VVn = sess.run(model.Vn,feed_dict = val_dict)
Vn = np.reshape(VVn,(batch_size*Nn*seq_len,K))
response = y_val_sub.flatten()

from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=2)
pca.fit(Vn)
Vn_low = pca.transform(Vn)
plt.scatter(Vn_low[:,0],Vn_low[:,1],c=response)