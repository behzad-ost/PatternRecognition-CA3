import FeatureSelection as fs
import numpy as np
import time
import prettytable

class KnnPdfClassifier:
	def __init__(self, k, train_set, test_set):
		self.train_set = train_set
		self.test_set = test_set
		self.k = k

		self.confusion = np.zeros([10,10])
		self.confidence = np.zeros([10,10])

		self.train_classified = [[] for i in range(10)]
		for label, data in zip(fs.train_labels, fs.train_data):
			self.train_classified[label].append(data)
		self.priors= [len(self.train_classified[n])/float(len(train_set[0])) for n in range(10)]
		self.d=62

	def knn_estimation (self, x_samples, x):
	    distances=[]
	    for x_sample in x_samples:
	    	r = np.sqrt(np.sum((x-x_sample)**2))
	        distances.append(r)
	    distances = np.sort(distances)
	    return distances[self.k-1]

	def classify(self, sample, true_label):
		results = [self.knn_estimation(self.train_classified[y],sample) for y in range(10)]
		label = np.argmin(results)
		
		sorted_args = np.argsort(results)
		max_prob = results[sorted_args[0]]
		second_max_prob = results[sorted_args[1]]
		
		self.confusion[true_label][label] = self.confusion[true_label][label] + 1
		self.confidence[true_label][label] += (second_max_prob - max_prob)

		return label

	def get_conf_matrixes(self):
		return self.confusion, self.confidence

def save_data(train_time, test_time, ccr):
	times = prettytable.PrettyTable(["train_time", "test_time", "correct_classification_rate"])
	times.add_row([train_time, test_time, ccr])
	

def mnist_digit_recognition(k=3):
	test_time = 0
	knn_classifier = KnnPdfClassifier(k, [fs.train_data,fs.train_labels], [fs.test_data,fs.test_labels])
	limit = len(fs.test_data)
	# limit = 10
	test_data, test_labels = fs.test_data[:limit], fs.test_labels[:limit]
	results = np.arange(limit, dtype=np.int)
	test_start_time = time.time()
	for n in range(limit):
		results[n] = knn_classifier.classify(test_data[n], test_labels[n])
		# print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])
	test_time = time.time() - test_start_time

	confusion, confidence =  knn_classifier.get_conf_matrixes()
	np.savetxt('matrixes/5.2.conusion.'+str(k), confusion, fmt="%3d")
	np.savetxt('matrixes/5.2.confidence'+str(k), confidence, fmt="%1.5f")

	print "K: ", k, "---> ","recognition rate: ", (results == test_labels).mean()
	return (results == test_labels).mean() , test_time


if __name__=="__main__":
	results = prettytable.PrettyTable(["k", "correct_classification_rate", "test_time" , "train_time"])
	for k in range(5):
		ccr, test_t = mnist_digit_recognition(k+1)
		results.add_row([k+1, ccr, test_t, 0])

	print(results)
	data = results.get_string()
	with open('knn.data', 'wb') as f:
		f.write(data)


