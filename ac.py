import tensorflow as tf
from dataset import *
import numpy as np
import random
import matplotlib.pyplot as plot
import pickle

data = load_dataset('data')
random.shuffle(data)

input_processed_n = 512
hidden_m = 128
hidden_k = 32
learn_rate = 0.014
train_steps = 100
train_output_folder = 'train-ac/'

bla = 0

def my_convolve(a):

	global bla

	c = [0] * len(a)
	for i in range(len(a)):

		b = np.fft.fft(a[i] * bh_window)
		b = b * np.conj(b)
		b = np.real(np.fft.ifft(b))

		# Find the high (excluding the first 44)
		hid = np.argmax(b[40:]) + 40
		b = resample(b, len(b)*input_processed_n // hid)[:len(b)]

		c[i] = b[: input_processed_n]

		
	c = np.array(c)

	#print(c.shape)
	#print(c.dtype)
	#bla += 1
	#plot.plot(c[bla])
	#plot.show()
	#plot.plot(a[bla][: input_processed_n])
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

# TRY RELU AND SIGMOID

# Layer outputs, tanh activation
# Tried. They suck.
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

	train_str = train_output_folder + '/' + training_id

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

	# data_train = reduce_dataset(data_train, 0.1)
	# data_test = reduce_dataset(data_test, 0.1)

	##################################################

	print(len(data_train))

	arr_train, tgt_train = prepare(data_train)
	arr_test, tgt_test = prepare(data_test)

	convd_arr_train = my_convolve(arr_train)
	convd_arr_test = my_convolve(arr_test)

	# Train

	for step in range(train_steps):

		whatever, loss, pred = sess.run([optimizer, cost, y], feed_dict={
			x: convd_arr_train,
			t: tgt_train
		})

		correct, total = calc_accuracy(pred, tgt_train)
		print("loss: {} step: {} / {} accuracy: {} / {}".format(
			loss, step, train_steps, correct, total))

		# Test after every step

		pred = sess.run(y, feed_dict={
			x: convd_arr_test,
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
			out_b3=out_b3,
		)

	with open(train_str + '.info', 'wb') as fw:
		pickle.dump([
			input_n,
			input_processed_n,
			hidden_m,
			hidden_k,
		], fw)

	# Test

	pred = sess.run(y, feed_dict={
		x: convd_arr_test,
		t: tgt_test
	})

	correct, total = calc_accuracy(pred, tgt_test)	

	print('{} / {} correct'.format(correct, total))

print('done.')