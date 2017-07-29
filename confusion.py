import numpy as np
from os import sep
import pickle
import pyaudio
from dataset import *
import sys

train_id = '9999383318195b73'
train_id = '35f9c523a55c3242'
train_id = '3443e93797531617'

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

def run_on_file(filename):
	dataset_file = load_file(filename)
	n = len(dataset_file)

	datasets = [dataset_file]

	for dataset in datasets:
		dd = {}
		carr = my_convolve(dataset)
		for i in range(len(carr)):
			gg = guess(carr[i])
			gl = 'aeiou'[np.argmax(gg)]
			if not gl in dd:
				dd[gl] = 1
			else:
				dd[gl] += 1
		h = 0
		y = None
		for x in dd:
			if dd[x] > h:
				y = x
				h = dd[x]
		sol.append(y)

	return ''.join(sol)

if __name__ == '__main__':
	data = load_dataset('data')
	data = split_actors_dataset(data)[1]
	data, tgts = prepare(data)
	data = my_convolve(data)
	matrix = np.zeros((5, 5), dtype=np.int32)

	for i in range(len(data)):
		gv = guess(data[i])
		gl = np.argmax(gv)
		tv = tgts[i]
		tl = np.argmax(tv)

		matrix[tl][gl] += 1

	print(matrix)