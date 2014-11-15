import numpy
import util

class BasicNetwork:
	def __init__(self, shape):
		# shape is an iterable of integers, each representing a layer
		# each integer specifies the number of neurons in that layer
		self.layers = len(shape)
		self.shape = shape
		self.weights = []
		self.bias = []
		for i in xrange(len(shape) - 1):
			# randomly generate matrix of weights for current layer
			# W_jk = weight of neuron k in previous layer (i) for neuron j in current layer (i+1)
			prev_count = shape[i]
			cur_count = shape[i + 1]
			self.weights.append(numpy.random.rand(cur_count, prev_count) * 2 - 1)

			# also generate a bias vector, b_j = bias for neuron j in current layer
			self.bias.append(numpy.random.rand(cur_count, 1) * 2 - 1)

	def forward(self, inputs, return_activations = False):
		# inputs specifies one float for each neuron in the first layer
		# if return_activations is true, we return a list of activations at each layer
		#  otherwise, we only return the output layer
		values = numpy.array([inputs], float).T # values in current layer
		activations = [values]

		for i in xrange(len(self.weights)):
			# compute the weighted sum of neuron inputs, plus bias term (for each neuron in current layer)
			z_vector = numpy.dot(self.weights[i], values) + self.bias[i]
			# apply sigmoid activation function
			values = util.sigmoid(z_vector)
			if return_activations: activations.append(values[0, :])

		if return_activations: return activations
		else: return values[0, :]

	def backward(self, inputs, desired_outputs):
		# inputs is a list of input layer values, desired_outputs is expected output layer values
		inputs = numpy.array([inputs], float).T
		desired_outputs = numpy.array([desired_outputs], float).T

		# execute forward propogation and store activations of each layer
		activations = self.forward(inputs[:, 0], True)

		# compute deltas at each layer
		deltas_list = [None] * self.layers # deltas_list[0] is for input layer and remains None
		deltas_list[self.layers - 1] = numpy.multiply(activations[self.layers - 1] - desired_outputs, util.sigmoid_d2(activations[self.layers - 1]))

		for l in xrange(self.layers - 2, 0, -1): # for each non-input layer
			previous_deltas = deltas_list[l + 1]
			sums = numpy.dot(self.weights[l].T, previous_deltas.reshape(len(previous_deltas), 1))
			deltas_list[l] = numpy.multiply(sums, util.sigmoid_d2(activations[l]))

		# compute squared-error partial derivatives with respect to weight and bias parameters
		# we do this in a list indexed by layer, and then in an array
		weight_derivatives = []
		bias_derivatives = []
		for l in xrange(self.layers - 1): # loop over each weights
			weight_derivatives.append(numpy.multiply(deltas_list[l + 1], activations[l].T))
			bias_derivatives.append(deltas_list[l + 1].reshape(len(deltas_list[l + 1]), 1))

		return (weight_derivatives, bias_derivatives)

	def gradient_iteration(self, x, y, learning_rate = 5.0, weight_decay = 0.):
		# x is a 2D array, each row is an input
		# y is a 2D array, each row is corresponding output

		# initialize sum of squared-error partial derivatives
		weight_d_sum = []
		bias_d_sum = []

		for l in xrange(len(self.weights)):
			weight_d_sum.append(numpy.zeros(self.weights[l].shape))
			bias_d_sum.append(numpy.zeros(self.bias[l].shape))

		# sum over each data point
		for i in xrange(len(x)):
			weight_d_i, bias_d_i = self.backward(x[i], y[i])

			for l in xrange(len(self.weights)):
				weight_d_sum[l] += weight_d_i[l]
				bias_d_sum[l] += bias_d_i[l]

		# update parameters
		for l in xrange(len(self.weights)):
			weight_update = numpy.multiply(1 / float(len(x)), weight_d_sum[l]) + weight_decay * self.weights[l]
			self.weights[l] -= learning_rate * weight_update

			bias_update = numpy.multiply(1 / float(len(x)), bias_d_sum[l])
			self.bias[l] -= learning_rate * bias_update

	def gradient(self, x, y, it = 1000, learning_rate = 5.0, weight_decay = 0.):
		for i in xrange(it):
			print i
			self.gradient_iteration(x, y, learning_rate, weight_decay)

import random
network = BasicNetwork((2, 2, 1))
x = []
y = []
for i in xrange(100):
	x_i = [random.random(), random.random()]
	x.append(x_i)
	if x_i[0] > 0.5 and x_i[1] > 0.5: y.append(0.999)
	else: y.append(0)

network.gradient(x, y)
print network.forward([0.4, 0.5])
print network.forward([0.1, 0.3])
print network.forward([0.7, 0.8])
