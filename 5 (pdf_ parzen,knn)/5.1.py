import FeatureSelection as fs
import numpy as np
import time
import prettytable
from sklearn.metrics import confusion_matrix


class ParzenPdfClassifier:
	def __init__(self, raduis, train_set, test_set):
		self.train_set = train_set
		self.test_set = test_set
		self.raduis = raduis
		self.confusion = np.zeros([10,10])
		self.confidence = np.zeros([10,10])

		self.train_classified = [[] for i in range(10)]
		for label, data in zip(fs.train_labels, fs.train_data):
			self.train_classified[label].append(data)

		self.priors= [len(self.train_classified[n])/float(len(train_set[0])) for n in range(10)]

		self.d=62

	def kernel_func(self, x,x_i):
		return (x - x_i) / (self.raduis)

	def rect_window_func(self, x):
		for feature in x:
			if np.abs(feature) > (self.raduis/2.):
				return 0
		return 1

	def gauss_window_func(self, x):
		mu = [0 for i in range(self.d)]
		cov = np.eye(self.d)

		part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
		part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
		
		return float(part1 * np.exp(part2))

	def parzen_estimation(self, x_samples, x, window_func):
		k_n = 0
		for x_sample in x_samples:
			x_i = self.kernel_func(x=x, x_i=x_sample)
			k_n += window_func(x_i)
		return (k_n / float(len(x_samples))) / (self.raduis**self.d)


	def classify(self, sample, window_func, true_label):
		results = [self.parzen_estimation(self.train_classified[y],sample,window_func) for y in range(10)]
		label = np.argmax(results)

		sorted_args = np.argsort(results)
		max_prob = results[sorted_args[-1]]
		second_max_prob = results[sorted_args[-2]]
		
		self.confusion[true_label][label] = self.confusion[true_label][label] + 1
		self.confidence[true_label][label] += (max_prob - second_max_prob)

		return label

	def get_conf_matrixes(self):
		return self.confusion, self.confidence

def mnist_digit_recognition(raduis=1):
	st = time.time()
	pazen_classifier = ParzenPdfClassifier(raduis, [fs.train_data,fs.train_labels], [fs.test_data,fs.test_labels])
	limit = len(fs.test_data)
	# limit = 10
	test_data, test_labels = fs.test_data[:limit], fs.test_labels[:limit]
	results = np.arange(limit, dtype=np.int)
	results_gauss_window = np.arange(limit, dtype=np.int)
	for n in range(limit):
		results[n] = pazen_classifier.classify(test_data[n],pazen_classifier.rect_window_func,test_labels[n])
		results_gauss_window[n] = pazen_classifier.classify(test_data[n],pazen_classifier.gauss_window_func, test_labels[n])
		# print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])

	confusion, confidence = pazen_classifier.get_conf_matrixes()
	np.savetxt('matrixes/5.1.confidence.'+str(raduis),confidence,fmt="%3d" )
	np.savetxt('matrixes/5.1.conusion.'+str(raduis),confusion,fmt="%1.5f")

	et = time.time()

	test_time = et - st

	print "Raduis: ", raduis, "---> ","rect recognition rate: ", (results == test_labels).mean()
	print "Raduis: ", raduis, "---> ","gauss recognition rate: ", (results_gauss_window == test_labels).mean()
	return test_time, (results_gauss_window == test_labels).mean(), (results == test_labels).mean()


if __name__=="__main__":
	results = prettytable.PrettyTable(["radius", "rect ccr", "gauss ccr", "test_time", "train_time"])
	for r in range(10):
		ttime, res_gauss,res_rect = mnist_digit_recognition(1.00 + r/20.)
		results.add_row([1.00 + r/20., res_rect, res_gauss, ttime, 0])

	# print(results)
	data = results.get_string()
	with open('parzen.data', 'wb') as f:
		f.write(data)