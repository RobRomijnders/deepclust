# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 17:47:36 2016

@author: rob
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scikits.audiolab
from scipy import signal
#import sounddevice as sd


def preprocess(filename,low_freq=6000,tr=0.15):
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

filename1 = '/home/rob/Dropbox/ml_projects/deepclust/data/male1.wav'
filename2 = '/home/rob/Dropbox/ml_projects/deepclust/data/female1.wav'
fs1,X1 = preprocess(filename1)
fs2,X2 = preprocess(filename2)

print('Signal1 has %.1f seconds left'%(len(X1)/fs1))
print('Signal2 has %.1f seconds left'%(len(X2)/fs2))

start = 1000000
Xs1 = X1[start:start+10*fs1]
Xs2 = X2[start:start+10*fs2]

if False:
  scikits.audiolab.play(0.025*Xr1, fs=fs2)
if False:
  scikits.audiolab.play(0.05*(Xs1+Xs2), fs=fs2)
#sd.play(X, fs)


#Plot Spectrogram
plt.figure()
f1, t1, Sxx1 = signal.spectrogram(Xs1, fs1,window=('hann'),nperseg = int(0.032*fs1))
f2, t2, Sxx2 = signal.spectrogram(Xs2, fs2,window=('hann'),nperseg = int(0.032*fs2))
fc, tc, Sxxc = signal.spectrogram(Xs1+Xs2, fs1,window=('hann'),nperseg = int(0.032*fs1))

Sxx1l = np.log(Sxx1)
Sxx2l = np.log(Sxx2)
Sxxcl = np.log(Sxxc)

bw = ((Sxx1l>Sxx2l).astype('float'))*2.0-1.0



plt.pcolormesh(tc, fc, Sxxcl*bw,cmap='bwr')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#Plot twenty random STFT
row = 10
ind = np.random.choice(Sxxcl.shape[1],row)
f, axarr = plt.subplots(row, 1)
for r in range(row):
  for p in range(len(Sxxcl[:,ind[r]])):
    if bw[p,ind[r]]>0.0:
      axarr[r].plot(p,Sxxcl[p,ind[r]],'r_')
    else:
      axarr[r].plot(p,Sxxcl[p,ind[r]],'b_')


"""Generate our own spectogram"""
Xr1 = np.zeros_like(Xs1)
Xrh1 = np.zeros_like(Xs1)
M = 280
R = 70
N = len(Xs1)

bins = np.floor(N/R)

t = np.linspace(0,N/fs1,bins)
f = np.linspace(0,fs1/2,M/2)
Slog = np.zeros((M/2,bins))
for s,i in enumerate(range(0,N-M-1,R)):
  #Reconstruct the signal
  x1 = np.hamming(M)*Xs1[i:i+M]
  x2 = np.hamming(M)*Xs2[i:i+M]

  #Construct spectogram
  x1_fft = np.fft.fft(x1)
  x2_fft = np.fft.fft(x2)
  high = (np.abs(x1_fft)>np.abs(x2_fft)).astype('float')
  Slog[:,s] = np.log(np.abs(x1_fft[:M/2]))

  x1_ifft = np.fft.ifft(x1_fft)
  x1_iffth = np.fft.ifft(high*x1_fft)

  Xr1[i:i+M] += np.real(x1_ifft)
  Xrh1[i:i+M] += np.real(x1_iffth)

plt.pcolormesh(t, f, Slog,cmap='Blues')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
