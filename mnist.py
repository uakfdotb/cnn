import network
import gzip
import struct
import numpy

def readImages(fname):
	images = []

	f = gzip.open(fname, 'rb')
	f.read(4) # magic number
	num_images, rows, cols = struct.unpack('>III', f.read(12))
	print '[minst] readImages: reading %d images of %dx%d' % (num_images, rows, cols)
	image_format = '>' + 'B' * (rows * cols)

	for i in xrange(num_images):
		images.append(numpy.array(struct.unpack(image_format, f.read(rows * cols)), float) / 256.)

		if i % 1000 == 0:
			print '[minst] ... read %d/%d images' % (i, num_images)

	f.close()
	print '[minst] ... done'
	return (num_images, rows, cols), images

def readLabels(fname, cls = None):
	labels = []

	f = gzip.open(fname, 'rb')
	f.read(4) # magic number
	num_items = struct.unpack('>I', f.read(4))[0]
	print '[minst] readLabels: reading %d labels' % (num_items)

	count_high = 0

	for i in xrange(num_items):
		label = struct.unpack('B', f.read(1))[0]
		if cls is None:
			y = [0.] * 10
			y[label] = 1.
			labels.append(y)
		else:
			if label == cls:
				labels.append(1.)
				count_high +=  1
			else:
				labels.append(0.)

	f.close()
	print '[minst] ... done (%d)' % count_high
	return labels

meta, x = readImages('minst/train-images-idx3-ubyte.gz')
y = readLabels('minst/train-labels-idx1-ubyte.gz')
cnn = network.BasicNetwork((len(x[0]), {'type': 'convsample', 'm': meta[1], 'c': 1, 'n': 7, 'k': 32, 'p': 11}, {'type': 'softmax', 'count': 10}))
cnn.sgd(x, y, 30)

meta, x = readImages('minst/t10k-images-idx3-ubyte.gz')
y = readLabels('minst/t10k-labels-idx1-ubyte.gz')
errors = 0

for i in xrange(len(x)):
	test_label = numpy.argmax(cnn.forward(x[i]))
	#print y[i], cnn.forward(x[i]), test_label

	if y[i][test_label] < 0.5:
		errors += 1
print 'errors: %d' % errors
