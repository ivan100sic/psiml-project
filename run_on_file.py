import numpy as np
from os import sep
import pickle
import pyaudio
from dataset import *
import sys

train_id = '564e1e42afe613d3'

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

def run_on_file(filename, subdivisions):
	dataset_file = load_file_simple(filename)
	n = len(dataset_file)

	datasets = []

	for i in range(subdivisions):
		start = n*i // subdivisions
		end = n*(i+1) // subdivisions

		datasets.append(dataset_file[start:end])

	# Run each dataset chunk through the NN

	sol = []

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
	filenames = sys.argv[1:]
	for fn in filenames:
		print('file: {} vowels: {}'.format(fn, run_on_file(fn, 25)))
