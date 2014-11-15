import numpy

def sigmoid(m):
	return 1. / (1. + numpy.exp(-m))

def sigmoid_d2(sigmoid_output):
	# returns derivative of sigmoid function given the sigmoid function output
	return sigmoid_output * (1 - sigmoid_output)
