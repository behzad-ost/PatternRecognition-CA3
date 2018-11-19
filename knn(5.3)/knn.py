import numpy as np
import math
import FeatureSelection as fs 
import time
import prettytable
from sklearn.metrics import confusion_matrix

train_set = [fs.train_data, fs.train_labels]
test_set = [fs.test_data, fs.test_labels]

def take_second(item):
    return item[1]

class KNN:
	def __init__(self, k, train_set, test_set):
		self.train_set = train_set
		self.test_set = test_set
		self.k = k

		self.confusion = np.zeros([10,10])
		self.confidence = np.zeros([10,10])

	def distance_func(self, data1, data2):
		distance = 0
		if(len(data1) != len(data2)):
			raise ValueError("Cant Find Distance of datas without Same Length");
		for x in range(len(data1)):
			distance += pow((data1[x] - data2[x]), 2)
		return math.sqrt(distance)

	def find_neighbours(self, x):
		distances = []
		for i in range(len(self.train_set[0])):
			dist = self.distance_func(x, self.train_set[0][i])
			distances.append((train_set[1][i], dist))
		distances.sort(key=take_second)
		neighbors = []

		for i in range(self.k):
			neighbors.append(distances[i])
		return neighbors

	def choose_class(self, neighbors, true_label):
		class_votes = {}
		for i in range(len(neighbors)):
			response = neighbors[i][0]
			if response in class_votes:
				class_votes[response] += 1
			else:
				class_votes[response] = 1

		sorted_votes = sorted(class_votes.iteritems(), key=take_second, reverse=True)
		label = sorted_votes[0][0]
		self.confusion[true_label][label] = self.confusion[true_label][label] + 1
		pmax = sorted_votes[0][1] / float(len(neighbors))
		pcon = 0
		if len(sorted_votes) > 1:
			pcon = sorted_votes[1][1] / float(len(neighbors)) 
		self.confidence[true_label][label] += (pmax - pcon)
		return label

	def classify(self, x, true_label):
		neighbors = self.find_neighbours(x)
		return self.choose_class(neighbors, true_label)

	def get_conf_matrixes(self):
		return self.confusion, self.confidence


def mnist_digit_recognition(k=3):
	limit = len(fs.test_data)
	# limit = 10
	test_data, test_labels = fs.test_data[:limit], fs.test_labels[:limit]
	k = k
	knn = KNN(k, train_set, test_set)
	results = np.arange(limit, dtype=np.int)
	test_start_time = time.time()
	for n in range(limit):
		results[n] = knn.classify(fs.test_data[n], fs.test_labels[n])
		#print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])
	test_time = time.time() - test_start_time
	confusion, confidence =  knn.get_conf_matrixes()

	np.savetxt('matrixes/5.3.conusion.'+str(k), confusion, fmt="%3d")
	np.savetxt('matrixes/5.3.confidence'+str(k), confidence, fmt="%1.5f")

	print "recognition rate: ", (results == test_labels).mean()
	return (results == test_labels).mean() , test_time

if __name__=="__main__":
	results = prettytable.PrettyTable(["k", "correct_classification_rate", "test_time" , "train_time" ])
	for k in [1,3,5,10]:
		res, t_time = mnist_digit_recognition(k)
		results.add_row([k, res, t_time, 0])

	print(results)
	data = results.get_string()
	with open('data.txt', 'wb') as f:
		f.write(data)

