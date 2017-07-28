"""
	Creates datasets
"""

import numpy as np
from wav2np import read_wav
import glob
from scipy.signal import resample as scipy_resample
import random
from os import sep
import copy

# Global dataset settings

input_n = 1024

letters = {
	'a': 0,
	'e': 1,
	'i': 2,
	'o': 3,
	'u': 4
}

output_l = len(letters)

"""
	Returns a python array representing all samples
	from the specified folder.

	Files should be named [aeiou]-[mf]-[0-9]+.wav
"""

def load_file(file):
	rate, raw = read_wav(file)
	raw = np.array(raw, dtype=np.float32)
	raw /= 32768.0
	regr_data_file_name = file.split('.')[0] + '.dat'
	file = file.split(sep)[-1]
	

	with open(regr_data_file_name) as f:
		for line in f:
			a = line.split(' ')
			frontness = 1 - float(a[0])
			openness = 1 - float(a[1])
			roundedness = float(a[2])

	letter = file.split('-')[0]
	sex = file.split('-')[1]
	actor = file.split('-')[2].split('.')[0]

	dataset = []

	separation = input_n // 32
	
	for i in range(input_n, len(raw), separation):
		block = np.array(raw[i - input_n : i])
		block -= np.sum(block) / len(block)
		block /= np.max(np.abs(block))
		dataset.append({
			'signal' : block,
			'letter' : letter,
			'sex': sex,
			'actor': actor,
			'frontness': frontness,
			'openness': openness,
			'roundedness': roundedness,
			'start': i-input_n,
			'end': i,
		})

	return dataset

def load_file_simple(file):
	rate, raw = read_wav(file)
	raw = np.array(raw, dtype=np.float32)
	raw /= 32768.0
	file = file.split(sep)[-1]
	
	dataset = []

	separation = input_n // 32
	
	for i in range(input_n, len(raw), separation):
		block = raw[i - input_n : i]
		dataset.append(block)

	return dataset

def load_dataset(folder):
	files = glob.glob(folder + sep + '*.wav')
	datasets = []
	for file in files:
		datasets += load_file(file)

	return datasets

def randomize_dataset(data):
	random.shuffle(data)


def reduce_dataset(data, fraction):
	ns = int(len(data) * fraction)
	return data[:ns]

def split_actors_dataset(data):

	actor_limit = '006'
	data_train = []
	data_test = []

	for datum in data:
		actor = datum['actor']
		if actor < actor_limit:
			data_train.append(datum)
		else:
			data_test.append(datum)

	return data_train, data_test

def noisify(data, scale=0.05):
	nudata = []
	for datum in data:
		narr = datum['signal']
		narr = narr + np.random.normal(0, scale, narr.shape[0])
		ndict = copy.copy(datum)
		ndict['signal'] = narr
		nudata.append(ndict)
	return nudata

def resample(arr, len):
	return scipy_resample(arr, len)

def pitch_shift(data, max_dev=128):
	nudata = []
	for datum in data:
		narr = datum['signal']
		nu_len = random.randint(input_n, input_n+max_dev)
		narr = resample(narr, nu_len)[:input_n]
		ndict = copy.copy(datum)
		ndict['signal'] = narr
		nudata.append(ndict)
	return nudata

def prepare(data):

	arr2 = []
	tgt2 = []

	for datum in data:
		arr = datum['signal']
		tgt = [0.0] * output_l
		tgt[letters[datum['letter']]] = 1.0

		arr2.append(arr)
		tgt2.append(tgt)

	return arr2, tgt2

def prepare_regr(data):

	arr2 = []
	tgt2 = []
	vow2 = []

	for datum in data:
		arr = datum['signal']
		tgt = [datum['frontness'], datum['openness'], datum['roundedness']]
		vow = datum['letter']

		arr2.append(arr)
		tgt2.append(tgt)
		vow2.append(vow)

	return arr2, tgt2, vow2

# Extracted from toy.py
#####################################

input_processed_n = 512
freq_cutoff_offset = 0

ludolf = 3.14159265358979323
bhw_a0 = 0.3635819
bhw_a1 = 0.4891775
bhw_a2 = 0.1365995
bhw_a3 = 0.0106411
bhw_ls = np.linspace(0, 1, input_n)
bh_window = bhw_a0 - bhw_a1 * np.cos(2*ludolf*bhw_ls)
bh_window += bhw_a2 * np.cos(4*ludolf*bhw_ls)
bh_window -= bhw_a3 * np.cos(6*ludolf*bhw_ls)
#energy = np.arange(freq_cutoff_offset,
#	freq_cutoff_offset+input_processed_n)

def my_convolve(a):

	c = [0] * len(a)
	for i in range(len(a)):
		#c[i] = np.convolve(b[i] * hann, np.array([1, 1, 1, 1, 1, 1]),
		#	'same')[:input_processed_n]
		b = np.array(np.abs(np.fft.fft(a[i] * bh_window)), dtype=np.float32)
		# b = np.convolve(b, 'same')
		c[i] = b[freq_cutoff_offset : input_processed_n+freq_cutoff_offset]
		# c[i] *= energy

	c = np.array(c)

	#print(c.shape)
	#print(c.dtype)
	#bla += 1
	#plot.plot(c[bla])
	#plot.show()
	#raise Exception()

	return c

#################################3
# END OF EXTRACTED

if __name__ == "__main__":
	# test
	some_data = load_dataset('data')
	print(len(some_data))
	
	"""
	for unit in some_data:
		print(len(unit['signal']), unit['letter'], unit['sex'])
	"""

