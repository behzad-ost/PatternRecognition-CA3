import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import FeatureSelection as fs 
import time
import prettytable
from sklearn.metrics import confusion_matrix

if __name__=="__main__":
	limit = len(fs.test_data)
	# limit = 100
	test_data, test_labels = fs.test_data[:limit], fs.test_labels[:limit]
	k = 10
	knn=KNeighborsClassifier()
	train_st = time.time()
	knn.fit(fs.train_data , fs.train_labels)
	train_time = time.time() - train_st
	results = np.arange(limit, dtype=np.int)
	test_st = time.time()
	for n in range(limit):
		results[n] = knn.predict([fs.test_data[n]])
		print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])

	test_time = time.time() - test_st
	print "recognition rate: ", (results == test_labels).mean()

	resultstable = prettytable.PrettyTable(["correct_classification_rate", "test_time", "train_time"])
	resultstable.add_row([(results == test_labels).mean(), test_time, train_time])
	confusion = confusion_matrix(test_labels, results, labels=[0,1,2,3,4,5,6,7,8,9])
	np.savetxt('matrixes/knn.conusion', confusion, fmt="%3d")

	data = resultstable.get_string()
	with open('knn.data', 'wb') as f:
		f.write(data)