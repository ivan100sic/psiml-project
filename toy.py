import tensorflow as tf
from dataset import *
import numpy as np
import random
import matplotlib.pyplot as plot
import pickle
from os import sep

data = load_dataset('data')
random.shuffle(data)

input_processed_n = 512
freq_cutoff_offset = 0
hidden_m = 128
hidden_k = 32
learn_rate = 0.014
train_steps = 500
train_output_folder = 'train' + sep

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

def calc_accuracy(pred, tgt):

	correct = 0

	for i in range(len(pred)):
		j = np.argmax(pred[i])
		k = np.argmax(tgt[i])

		if j == k:
			correct += 1

	return correct, len(pred)

# x is input (processed block)
# t is the expected target

x = tf.placeholder(tf.float32, shape=(None, input_processed_n))
t = tf.placeholder(tf.float32, shape=(None, output_l))
# Layer 1 variables
W1 = tf.Variable(tf.random_normal([input_processed_n, hidden_m]),
	trainable=True, name='W1')
b1 = tf.Variable(tf.random_normal([hidden_m]),
	trainable=True, name='b1')
# Layer 2 variables
W2 = tf.Variable(tf.random_normal([hidden_m, hidden_k]),
	trainable=True, name='W2')
b2 = tf.Variable(tf.random_normal([hidden_k]),
	trainable=True, name='b2')
# Output layer variables
W3 = tf.Variable(tf.random_normal([hidden_k, output_l]),
	trainable=True, name='W3')
b3 = tf.Variable(tf.random_normal([output_l]),
	trainable=True, name='b3')

# Layer outputs, tanh activation
layer_1 = tf.tanh(tf.add(tf.matmul(x, W1), b1))
layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, W2), b2))
y = tf.nn.softmax(tf.add(tf.matmul(layer_2, W3), b3))

cost = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t))

optimizer = tf.train.AdamOptimizer(
	learning_rate=learn_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:

	# training id, save state, etc
	training_id = ''.join(
		['0123456789abcdef'[random.randint(0, 15)]
		for x in range(16)]
	)

	train_str = train_output_folder + sep + training_id

	print('training_id: {}'.format(training_id))

	sess.run(init)

	data_train, data_test = split_actors_dataset(data)
	dt_n = noisify(data_train)
	dt_ps = pitch_shift(dt_n)

	data_train += dt_n
	data_train += dt_ps

	randomize_dataset(data_train)
	randomize_dataset(data_test)

	#################### REMOVE ######################

	data_train = reduce_dataset(data_train, 1)
	data_test = reduce_dataset(data_test, 1)

	##################################################

	print(len(data_train))

	arr_train, tgt_train = prepare(data_train)
	arr_test, tgt_test = prepare(data_test)

	# Train

	for step in range(train_steps):

		whatever, loss, pred = sess.run([optimizer, cost, y], feed_dict={
			x: my_convolve(arr_train),
			t: tgt_train
		})

		correct, total = calc_accuracy(pred, tgt_train)
		print("step: {} / {} accuracy: {} / {} loss: {}".format(
			step, train_steps, correct, total, loss))

		# Test after every step

		pred = sess.run(y, feed_dict={
			x: my_convolve(arr_test),
			t: tgt_test
		})
		correct, total = calc_accuracy(pred, tgt_test)	
		print('{} / {} correct'.format(correct, total))

	# Save as file

	print('Saving...')

	out_W1, out_b1, out_W2, out_b2, out_W3, out_b3 = sess.run(
		[W1, b1, W2, b2, W3, b3])

	np.savez(train_str,

			out_W1=out_W1,
			out_b1=out_b1,
			out_W2=out_W2,
			out_b2=out_b2,
			out_W3=out_W3,
			out_b3=out_b3
		)

	with open(train_str + '.info', 'wb') as fw:
		pickle.dump([
			input_n,
			input_processed_n,
			freq_cutoff_offset,
			hidden_m,
			hidden_k,
		], fw)

	# Test

print('done.')