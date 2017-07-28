import numpy as np
from os import sep
import pickle
import pyaudio
from dataset import *

train_id = '9999383318195b73'
train_id = '35f9c523a55c3242'

train_data = np.load('train' + sep + train_id + '.npz')

W1 = train_data['out_W1']
W2 = train_data['out_W2']
W3 = train_data['out_W3']

b1 = train_data['out_b1']
b2 = train_data['out_b2']
b3 = train_data['out_b3']

def guess(x):
	y = np.tanh(np.matmul(x, W1) + b1)
	y = np.tanh(np.matmul(y, W2) + b2)
	y = np.matmul(y, W3) + b3

	y = np.exp(y)
	y /= y.sum()

	# y is a 5-vector of probabilities
	# print? process some more?

	return y

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

def record():

	# Credits: https://gist.github.com/mabdrabo/8678538

	
	
	data = stream.read(CHUNK)

	samples = []

	for i in range(0, 2048, 2):
		x = int(data[i]) + int(data[i+1]) * 256
		if x > 32767:
			x -= 65536

		samples.append(x)

	arr = np.array(samples, dtype=np.float32)
	arr -= arr.sum() / len(arr)
	arr /= np.max(np.abs(arr))

	return arr

print('Press enter to start recording...')
input()
while True:
	x = record()
	xc = my_convolve([x])
	y = guess(xc)[0]

	i = np.argmax(y)

	print('aeiou'[i], y[i] * 100)
