"""
	Creates datasets
"""

import numpy as np
from wav2np import read_wav
import glob
from scipy.signal import resample as scipy_resample
import random

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
	file = file.split('/')[-1]

	letter = file.split('-')[0]
	sex = file.split('-')[1]
	actor = file.split('-')[2].split('.')[0]

	dataset = []

	separation = input_n // 32
	
	for i in range(input_n, len(raw), separation):
		block = raw[i - input_n : i]
		dataset.append({
			'signal' : block,
			'letter' : letter,
			'sex': sex,
			'actor': actor,
		})

	return dataset

def load_dataset(folder):
	files = glob.glob(folder + '/*.wav')
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
		nudata.append({
			'signal' : narr,
			'letter' : datum['letter'],
			'sex': datum['sex'],
			'actor': datum['actor'],
		})
	return nudata

def resample(arr, len):
	return scipy_resample(arr, len)

def pitch_shift(data, max_dev=128):
	nudata = []
	for datum in data:
		narr = datum['signal']
		nu_len = random.randint(input_n, input_n+max_dev)
		narr = resample(narr, nu_len)[:input_n]
		nudata.append({
			'signal' : narr,
			'letter' : datum['letter'],
			'sex': datum['sex'],
			'actor': datum['actor'],
		})
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

if __name__ == "__main__":
	# test
	some_data = load_dataset('data')
	print(len(some_data))
	
	"""
	for unit in some_data:
		print(len(unit['signal']), unit['letter'], unit['sex'])
	"""

