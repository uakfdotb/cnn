import numpy
import scipy.signal
import time
import util

class BasicNetwork:
	def __init__(self, shape):
		# shape is an iterable of integers, each representing a layer
		# each integer specifies the number of neurons in that layer
		self.nlayers = len(shape)
		self.layers = []
		self.shape = list(shape)

		# first layer is input layer
		self.layers.append({'type': 'input'})

		# process other layers
		for i in xrange(len(shape) - 1):
			if isinstance(shape[i + 1], dict):
				if shape[i + 1]['type'] == 'convsample':
					# add a 2D convolutional/subsampling layer
					m = shape[i + 1]['m'] # input sidelength
					c = shape[i + 1]['c'] # number of channels in image
					n = shape[i + 1]['n'] # filter sidelength
					k = shape[i + 1]['k'] # number of convolutional filters
					p = shape[i + 1]['p'] # pooled region length
					conv_count = m - n + 1 # convolution output sidelength

					if self.shape[i] != m * m * c: # output size of previous layer must match image dimensions
						raise Exception('mismatching sidelength')

					if conv_count % p != 0: # convolution output must be divisible by pooled region size
						raise Exception('conv_count % p != 0')

					o = conv_count / p # number of pooled regions sidelength (also output sidelength)

					weights = numpy.random.rand(k * c, n, n) * 6 - 3
					bias = numpy.random.rand(k * c) * 2 - 1
					self.shape[i + 1] = k * o * o
					self.layers.append({'type': 'convsample', 'weights': weights, 'bias': bias, 'm': m, 'c': c, 'n': n, 'k': k, 'p': p, 'conv_count': conv_count, 'o': o})
				elif shape[i + 1]['type'] == 'softmax':
					# randomly generate matrix of weights for current layer
					# W_jk = weight of neuron k in previous layer (i) for neuron j in current layer (i+1)
					prev_count = self.shape[i]
					cur_count = self.shape[i + 1]['count']
					weights = numpy.random.rand(cur_count, prev_count) * 2 - 1

					# also generate a bias vector, b_j = bias for neuron j in current layer
					bias = numpy.random.rand(cur_count, 1) * 2 - 1

					self.shape[i + 1] = cur_count
					self.layers.append({'type': 'softmax', 'weights': weights, 'bias': bias})
			else:
				# randomly generate matrix of weights for current layer
				# W_jk = weight of neuron k in previous layer (i) for neuron j in current layer (i+1)
				prev_count = self.shape[i]
				cur_count = self.shape[i + 1]
				weights = numpy.random.rand(cur_count, prev_count) * 2 - 1

				# also generate a bias vector, b_j = bias for neuron j in current layer
				bias = numpy.random.rand(cur_count, 1) * 2 - 1

				self.layers.append({'type': 'sigmoid', 'weights': weights, 'bias': bias})

	def forward(self, inputs, return_activations = False):
		# inputs specifies one float for each neuron in the first layer
		# if return_activations is true, we return a list of activations at each layer
		#  otherwise, we only return the output layer
		values = numpy.array([inputs], float).T # values in current layer
		activations = [{'activations': values}]

		for layer in self.layers[1:]:
			if layer['type'] == 'sigmoid':
				# compute the weighted sum of neuron inputs, plus bias term (for each neuron in current layer)
				z_vector = numpy.dot(layer['weights'], values) + layer['bias']

				# apply sigmoid activation function
				values = util.sigmoid(z_vector)
				if return_activations: activations.append({'activations': values[:, 0]})
			elif layer['type'] == 'softmax':
				# compute the weighted sum of neuron inputs, plus bias term (for each neuron in current layer)
				z_vector = numpy.dot(layer['weights'], values) + layer['bias']

				# apply softmax
				values = numpy.exp(z_vector - numpy.max(z_vector))
				values = values / numpy.sum(values)
				if return_activations: activations.append({'activations': values[:, 0]})
			elif layer['type'] == 'convsample':
				# carry out convolution to get convolved values
				convolved_out = numpy.zeros((layer['k'], layer['conv_count'], layer['conv_count']))
				for k in xrange(len(layer['weights'])):
					convolved_out[k] = scipy.signal.convolve2d(values.reshape(layer['m'], layer['m']), numpy.rot90(layer['weights'][k], 2), 'valid')
					convolved_out[k] = util.sigmoid(convolved_out[k] + layer['bias'][k])

				# pool the convolved features
				pooled = numpy.zeros((layer['k'], layer['o'], layer['o']))
				for k in xrange(layer['k']):
					for i in xrange(layer['o']):
						for j in xrange(layer['o']):
							pooled[k][i][j] = numpy.average(convolved_out[k, (i * layer['p']):((i + 1) * layer['p']), (j * layer['p']):((j + 1) * layer['p'])])
				values = pooled.reshape(util.prod(pooled.shape), 1)
				if return_activations: activations.append({'activations': values[:, 0], 'extra': convolved_out})

		if return_activations: return activations
		else: return values[:, 0]

	def backward(self, inputs, desired_outputs):
		# inputs is a list of input layer values, desired_outputs is expected output layer values
		inputs = numpy.array([inputs], float).T
		desired_outputs = numpy.array([desired_outputs], float).T

		# execute forward propogation and store activations of each layer
		activations = self.forward(inputs[:, 0], True)

		# compute deltas at each layer
		deltas_list = [None] * self.nlayers # deltas_list[0] is for input layer and remains None

		if self.layers[self.nlayers - 1]['type'] == 'sigmoid':
			cost = numpy.sum(abs(activations[self.nlayers - 1]['activations'] - desired_outputs[:, 0]))
			deltas_list[self.nlayers - 1] = numpy.multiply(activations[self.nlayers - 1]['activations'] - desired_outputs[:, 0], util.sigmoid_d2(activations[self.nlayers - 1]['activations']))
		elif self.layers[self.nlayers - 1]['type'] == 'softmax':
			cost = -numpy.sum(numpy.multiply(desired_outputs[:, 0], numpy.log(activations[self.nlayers - 1]['activations'])))
			deltas_list[self.nlayers - 1] = activations[self.nlayers - 1]['activations'] - desired_outputs[:, 0]
		else:
			raise Exception('invalid error function type')

		for l in xrange(self.nlayers - 2, 0, -1): # for each non-input layer
			previous_deltas = deltas_list[l + 1]
			target_layer = self.layers[l + 1]

			if target_layer['type'] == 'sigmoid' or target_layer['type'] == 'softmax':
				sums = numpy.dot(target_layer['weights'].T, previous_deltas.reshape(util.prod(previous_deltas.shape), 1))
				deltas_list[l] = numpy.multiply(sums.flatten(), util.sigmoid_d2(activations[l]['activations']))
			elif target_layer['type'] == 'convsample':
				#deltas_list[l] = numpy.zeros((target_layer['k'], activations[l].shape[0], activations[l].shape[1]))
				#previous_deltas = previous_deltas.reshape(target_layer['k'], target_layer['p'], target_layer['p'])

				#for k in xrange(target_layer['k']):
				#	pooled_size = target_layer['m'] / target_layer['p']
				#	sums = (1. / pooled_size / pooled_size) * ponumpy.kron(previous_deltas[k], numpy.ones((pooled_size, pooled_size))) # upsample over mean pooling
				#	deltas_list[l][k, :, :] = numpy.multiply(sums, util.sigmoid_d2(activations[l][k].flatten()))
				#	deltas_list[l] = None
				raise Exception('convsample bp')

		# compute squared-error partial derivatives with respect to weight and bias parameters
		# we do this in a list indexed by layer, and then in an array
		weight_derivatives = []
		bias_derivatives = []
		for l in xrange(self.nlayers - 1): # loop over each weights
			previous_deltas = deltas_list[l + 1]
			target_layer = self.layers[l + 1]

			if target_layer['type'] == 'sigmoid' or target_layer['type'] == 'softmax':
				weight_derivatives.append(numpy.dot(previous_deltas.reshape(len(previous_deltas), 1), activations[l]['activations'].reshape(1, len(activations[l]['activations']))))
				bias_derivatives.append(previous_deltas.reshape(len(deltas_list[l + 1]), 1))
			elif target_layer['type'] == 'convsample':
				previous_deltas = previous_deltas.reshape(target_layer['k'], target_layer['o'], target_layer['o'])
				delta_bp_pool = numpy.zeros((target_layer['k'], target_layer['conv_count'], target_layer['conv_count'])) # back-propogate error across pooling layer
				for k in xrange(target_layer['k']):
					delta_bp_pool[k] = numpy.kron(previous_deltas[k], numpy.ones((target_layer['p'], target_layer['p'])) / target_layer['p'] / target_layer['p'])
				delta_bp_conv = numpy.multiply(delta_bp_pool, util.sigmoid_d2(activations[l + 1]['extra']))

				# bias derivative from delta_bp_pool sum per filter
				bias_d = numpy.zeros(target_layer['k'])
				for k in xrange(target_layer['k']):
					bias_d[k] = numpy.sum(delta_bp_pool[k])
				bias_derivatives.append(bias_d)

				# weight derivative
				weight_d = numpy.zeros((target_layer['k'], target_layer['n'], target_layer['n']))
				for k in xrange(target_layer['k']):
					weight_d[k] = scipy.signal.convolve2d(activations[l]['activations'].reshape(target_layer['m'], target_layer['m']), numpy.rot90(delta_bp_conv[k], 2), 'valid')
				weight_derivatives.append(weight_d)

		return (weight_derivatives, bias_derivatives, cost)

	def gradient_iteration(self, x, y, learning_rate = 5.0, weight_decay = 0.):
		# x is a 2D array, each row is an input
		# y is a 2D array, each row is corresponding output

		# initialize sum of squared-error partial derivatives
		weight_d_sum = []
		bias_d_sum = []
		cost_sum = 0

		for layer in self.layers[1:]:
			weight_d_sum.append(numpy.zeros(layer['weights'].shape))
			bias_d_sum.append(numpy.zeros(layer['bias'].shape))

		# sum over each data point
		for i in xrange(len(x)):
			if i % 1000 == 0: print '[network] %.2f%%' % (float(i) / len(x) * 100)
			weight_d_i, bias_d_i, cost_i = self.backward(x[i], y[i])

			for l in xrange(self.nlayers - 1):
				weight_d_sum[l] += weight_d_i[l]
				bias_d_sum[l] += bias_d_i[l]

			cost_sum += cost_i

		# update parameters
		for l in xrange(self.nlayers - 1):
			weight_update = numpy.multiply(1 / float(len(x)), weight_d_sum[l]) + weight_decay * self.layers[l + 1]['weights']
			self.layers[l + 1]['weights'] -= learning_rate * weight_update

			bias_update = numpy.multiply(1 / float(len(x)), bias_d_sum[l])
			self.layers[l + 1]['bias'] -= learning_rate * bias_update

		return cost_sum

	def gradient(self, x, y, it = 1000, learning_rate = 5.0, time_limit = None, weight_decay = 0.):
		time_start = time.time()
		for i in xrange(it):
			cost = self.gradient_iteration(x, y, learning_rate, weight_decay)
			print i, cost

			if time_limit is not None and time.time() - time_start > time_limit:
				print 'stopping at iteration %d (elapsed: %d)' % (i, time.time() - time_start)
				break

	def sgd_iteration(self, x, y, iteration, weight_decay = 0.):
		# x is a 2D array, each row is an input
		# y is a 2D array, each row is corresponding output
		random_indices = numpy.random.choice(range(len(x)), 256)
		x_batch = x[random_indices]
		y_batch = y[random_indices]
		learning_rate = 5. / (50. + iteration)
		momentum = 0.5

		if iteration > 20:
			momentum = 0.95

		# initialize sum of squared-error partial derivatives
		weight_d_sum = []
		bias_d_sum = []
		cost_sum = 0

		for layer in self.layers[1:]:
			weight_d_sum.append(numpy.zeros(layer['weights'].shape))
			bias_d_sum.append(numpy.zeros(layer['bias'].shape))

		# sum over each data point
		for i in xrange(len(x_batch)):
			weight_d_i, bias_d_i, cost_i = self.backward(x_batch[i], y_batch[i])

			for l in xrange(self.nlayers - 1):
				weight_d_sum[l] += weight_d_i[l]
				bias_d_sum[l] += bias_d_i[l]

			cost_sum += cost_i

		for l in xrange(len(self.layers) - 1):
			if self.layers[l + 1]['type'] != 'convsample':
				weight_d_sum[l] /= float(len(x_batch))
				bias_d_sum[l] /= float(len(x_batch))

		# update parameters
		for l in xrange(self.nlayers - 1):
			self.velocity[l + 1]['weights'] = momentum * self.velocity[l + 1]['weights'] + learning_rate * (weight_d_sum[l] + weight_decay * self.layers[l + 1]['weights'])
			self.layers[l + 1]['weights'] -= self.velocity[l + 1]['weights']
			print learning_rate, self.velocity[l + 1]['weights']

			if self.layers[l + 1]['type'] != 'convsample':
				self.velocity[l + 1]['bias'] = momentum * self.velocity[l + 1]['bias'] + learning_rate * bias_d_sum[l]
				self.layers[l + 1]['bias'] -= self.velocity[l + 1]['bias']

		print '[network] sgd: completed iteration %d (cost=%.4f)' % (iteration, cost_sum)

	def sgd(self, x, y, it = 5000, time_limit = None, weight_decay = 0.):
		# convert to numpy arrays
		x = numpy.array(x, float)
		y = numpy.array(y, float)

		# initialize velocity
		self.velocity = [None]
		for l in xrange(self.nlayers - 1):
			velocity_weights = numpy.zeros(self.layers[l + 1]['weights'].shape)
			velocity_bias = numpy.zeros(self.layers[l + 1]['bias'].shape)
			self.velocity.append({'weights': velocity_weights, 'bias': velocity_bias})

		time_start = time.time()

		for i in xrange(it):
			self.sgd_iteration(x, y, i, weight_decay)

			if time_limit is not None and time.time() - time_start > time_limit:
				print 'stopping at iteration %d (elapsed: %d)' % (i, time.time() - time_start)
				break

if __name__ == "__main__":
	import random
	network = BasicNetwork((100, {'type': 'convsample', 'm': 10, 'c': 1, 'n': 5, 'k': 3, 'p': 3}, 1))
	# {'type': 'convsample', 'm': 10, 'c': 1, 'n': 5, 'k': 3, 'p': 3}
	x = []
	y = []
	for i in xrange(2000):
		x_i = numpy.random.rand(100) / 10
		if random.random() < 0.5:
			x_i[0] = random.random()
			x_i[11] = random.random()
			x_i[22] = random.random()
			x_i[33] = random.random()
			x_i[44] = random.random()
			y.append(0.999)
		else:
			x_i[10] = random.random()
			x_i[19] = random.random()
			x_i[27] = random.random()
			x_i[36] = random.random()
			x_i[45] = random.random()
			y.append(0)
		x.append(x_i.tolist())

	network.sgd(x, y, 200)
	print network.layers

	for i in xrange(30):
		x_i = numpy.random.rand(100) / 10
		if random.random() < 0.5:
			x_i[0] = random.random()
			x_i[11] = random.random()
			x_i[22] = random.random()
			x_i[33] = random.random()
			x_i[44] = random.random()
			print 'orange', network.forward(x_i)
		else:
			x_i[10] = random.random()
			x_i[19] = random.random()
			x_i[27] = random.random()
			x_i[36] = random.random()
			x_i[45] = random.random()
			print 'blue', network.forward(x_i)
