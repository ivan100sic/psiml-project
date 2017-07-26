"""
	Wav file I/O
"""

import numpy as np
from scipy.io.wavfile import read as s_wav_read
from scipy.io.wavfile import write as s_wav_write

"""
	Returns (sample rate, numpy array)
"""
def read_wav(filename):
	return s_wav_read(filename)

def write_wav(filename, rate, data):
	s_wav_write(filename, rate, data)

if __name__ == "__main__":
	
	# Run some tests
	filename = "test/noise.wav"

	arr = np.random.random_integers(-32768, 32767, (22050))
	arr = np.array(arr, dtype=np.int16)

	write_wav(filename, 44100, arr)

	filename = "data/a-m-000.wav"

	rate, arr = read_wav(filename)

	print(arr)
	print(rate)