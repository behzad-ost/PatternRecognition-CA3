import numpy as np
import scipy
import scipy.stats
import cPickle
import time
import prettytable
import FeatureSelection as fs 
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

class GaussianBayes:
	def __init__(self, n_labels, n_features, train_set, test_set):
		self.n_labels = np.array(n_labels)
		self.n_features = np.array(n_features)
		self.train_set = train_set
		self.test_set = test_set

		self.confusion = np.zeros([10,10])
		self.confidence = np.zeros([10,10])

		self.train_classified = [[] for i in range(10)]
		for label, data in zip(train_set[1], train_set[0]):
			self.train_classified[label].append(data)
		self.priors= [len(self.train_classified[n])/float(len(train_set[0])) for n in range(10)]

	def classify(self, data, true_label):
		results = [self.gauss_pdf(data, self.mu[y], self.cov[y], self.priors[y]) for y in range(self.n_labels)]
		# print "results"
		# print results

		label = np.argmax(results)

		sorted_args = np.argsort(results)
		max_prob = results[sorted_args[-1]]
		second_max_prob = results[sorted_args[-2]]
		
		self.confusion[true_label][label] = self.confusion[true_label][label] + 1
		self.confidence[true_label][label] += (np.log(max_prob) - np.log(second_max_prob))

		#Problem 4
		treshhold = 1 - 1 #0.8
		if results[label] > treshhold or results[label] == treshhold:
			res = label
		else:
			res = -1 #reject
		return res

	def get_conf_matrixes(self):
		return self.confusion, self.confidence

	def train(self):
		self.cov=[]
		self.mu=[]
		for train_class in self.train_classified:
			self.cov.append(np.cov(np.array(train_class).T))
			self.mu.append(np.mean(np.array(train_class),axis=0))  
		self.cov = np.array(self.cov)
		self.mu = np.array(self.mu)


	def get_parameters(self):
		return ([self.mean, self.var], self.pi)

	def gauss_pdf(self, x, mu, cov, prior):
		# part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
		# part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
		#float(part1 * np.exp(part2))

		p = multivariate_normal(mean=mu, cov=cov)
		return p.pdf(x)*prior


def save_data(train_time, test_time, ccr):
	times = prettytable.PrettyTable(["train_time", "test_time", "correct_classification_rate", "probability_of_error"])
	times.add_row([train_time, test_time, ccr, 1-ccr])
	data = times.get_string()
	with open('data.txt', 'wb') as f:
		f.write(data)

def mnist_digit_recognition():
	train_time = 0
	test_time = 0
	
	n_labels = 10
	n_features = fs.feature_size
	train_data = fs.train_data
	train_labels = fs.train_labels
	test_data = fs.test_data
	test_labels = fs.test_labels


	pca = PCA(n_components=50)
	pca.fit(train_data)
	train_d=pca.transform(train_data)
	train_d.shape
	test_d=pca.transform(test_data)
	train_set = [train_d, train_labels]
	test_set = [test_d, test_labels]

	mnist_model = GaussianBayes(n_labels, n_features, train_set, test_set)
	train_test_time = time.time()
	mnist_model.train()
	train_time = time.time() - train_test_time
	# slice
	limit = len(fs.test_data)
	# limit = 10
	test_data, test_labels = test_d[:limit], test_labels[:limit]
	results = np.arange(limit, dtype=np.int)
	test_start_time = time.time()
	for n in range(limit):
		results[n] = mnist_model.classify(test_data[n], test_labels[n])
		print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])

	test_time = time.time() - test_start_time

	confusion, confidence =  mnist_model.get_conf_matrixes()

	# print confusion_matrix(test_labels, results, labels=[0,1,2,3,4,5,6,7,8,9])
	# print "==============="
	# print confusion

	np.savetxt('matrixes/3.confidence', confidence, fmt="%1.5f")
	np.savetxt('matrixes/3.conusion', confusion, fmt="%3d")

	save_data(train_time, test_time,(results == test_labels).mean())

	print "recognition rate: ", (results == test_labels).mean()
		
if __name__=="__main__":
	mnist_digit_recognition()


# data = results.get_string()
# 	with open('parzen_raduis_window.table', 'wb') as f:
# 		f.write(data)