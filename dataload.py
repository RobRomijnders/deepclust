# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:07:50 2016

@author: rob
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal

def preprocess(filename,low_freq=8000,tr=0.15):
  """Function to preprocess the speech file.
  input:
  - filename: the location where the .wav file is located
  - low_freq: the lowest sampling frequency allowed
  output:
  - Signal, where the first and final 5% is deleted and the 15% most silence utterances are deleted"""
  from scipy.io import wavfile
  from scipy.signal import decimate
  assert filename[-4:] == '.wav', 'please provide a wav file'

  #Z-normalize data
  fs,X = wavfile.read(filename)
  X = X.astype('float')
  if len(X.shape)==2:
    X = np.mean(X,axis=1)
  X-= np.mean(X)
  X /= np.std(X)

  #Decimate data to lowest frequency higher than low_freq
  fac = int(np.floor(fs/low_freq))
  X = decimate(X,fac)
  fs/=fac

  #Remove quartile with lowest loudness
  N = len(X)
  M = int(0.25*fs)  #Check bits of a quarter second
  Xp = []
  Xg = []

  for i in range(0,N-M,M):
    norm = np.sum(np.square(X[i:i+M]))
    Xp.append(norm)
  th = np.percentile(Xp,tr)
  for l,norm in enumerate(Xp):
    if norm > th:
      Xg.append(X[l*M:(l+1)*M])

  Xg = np.hstack(Xg)

  #Chop off first and final 5%
  Xg = Xg[int(0.05*len(Xg)):int(0.95*len(Xg))]

  return fs,Xg

def plot_spec(t,f,S):
  plt.pcolormesh(t, f, S,cmap='Blues')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.show()

class DataLoad():
  def __init__(self,sl = 100,Nn = 140,low_freq=8000,th=0.15):
    """Initialize the instance
    input
    - sl: sequence length. Defaults to 100 as in original paper
    - Nn: number of time-frequency coefficients, defaults to 140 as in original paper
    - low_freq: the lowest sample frequency that is acceptable
    - th: which percentage of most silence quarter-seconds to remove"""
    assert Nn%2 == 0, 'Please provide a Nn that can be divided by two'
    self.data = np.zeros((1,sl,Nn))  #Tensor for the data in [num_samples, seq_len, number_of_tf_coefficients]
    self.target = np.zeros((1,sl,Nn))
    self.low_freq = low_freq
    self.Nn = Nn
    self.sl = sl
    self.th = th



  def read_data(self,filename,vis=False,normalize=True):
    assert isinstance(filename,list),'Please provde a list of filenames'
    assert isinstance(filename[0],tuple),'Please provide a list of tuples'
    L = len(filename)
    for l in range(L):
      fs1,X1 = preprocess(filename[l][0], self.low_freq,tr=self.th)
      fs2,X2 = preprocess(filename[l][1], self.low_freq,tr=self.th)
      M = self.Nn*2
      R = int(self.Nn/2)
      N = np.min((len(X1),len(X2)))
      feats = []
      maps = []
      if vis:
        bins = np.floor(N/R)
        t = np.linspace(0,N/fs1,bins)
        f = np.linspace(0,fs1/2,M/2)
        Slog = np.zeros((M/2,bins))

      for s,i in enumerate(range(0,N-M-1,R)):
        #Reconstruct the signal
        x1 = np.hamming(M)*X1[i:i+M]
        x2 = np.hamming(M)*X2[i:i+M]

        #Construct spectogram
        x1_fft = np.fft.fft(x1)
        x2_fft = np.fft.fft(x2)
        high = (np.abs(x1_fft)>np.abs(x2_fft)).astype('float')

        x_fft_sum = np.log(np.abs(x1_fft[:M/2]))+np.log(np.abs(x2_fft[:M/2]))
        feats.append(x_fft_sum)
        maps.append(high[:M/2])
        if vis:
          Slog[:,s] = np.log(np.abs(x1_fft[:M/2]))
      if vis:
        plot_spec(t,f,Slog)
        plt.figure()
        plt.hist(Slog.flatten(),bins=30)
        plt.show()
      feats = np.vstack(feats)
      maps = np.vstack(maps)
      if normalize:
        feats -= np.mean(feats)
        feats /= np.std(feats)

      N = int(np.floor(feats.shape[0]/self.sl))
      assert N>1, 'Please provide a WAV file of enough length'
      SL = N*self.sl
      FEATS = np.reshape(feats[:SL],(N,self.sl,self.Nn))
      MAPS = np.reshape(maps[:SL],(N,self.sl,self.Nn))

      self.data = np.concatenate((self.data,FEATS))
      self.target = np.concatenate((self.target,MAPS))
      print(self.data.shape,self.target.shape)

    return
  def strip_zero(self):
    self.data = self.data[1:]
    self.target = self.target[1:]
    return

  def return_data(self,ratio = 0.9):
    """Function to return a trainset and validation set"""
    assert ratio <= 1.0 and ratio >= 0.0, 'Provide a ratio between 0 and 1'
    N = self.data.shape[0]
    ind_cut = int(ratio*N)
    ind = np.random.permutation(N)
    D = {}
    D['X_train'] = self.data[ind[:ind_cut]]
    D['X_val'] = self.data[ind[ind_cut:]]
    D['y_train'] = self.target[ind[:ind_cut]]
    D['y_val'] = self.target[ind[ind_cut:]]

    return D
























