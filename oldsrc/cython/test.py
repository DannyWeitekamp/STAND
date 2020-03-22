import d_tree
import numpy as np
import timeit

from sklearn import tree
# a = np.asarray([[6,1],[6,1]],dtype=np.int32)

data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
],np.bool);

labels = np.asarray([3,1,1,1,2,2,2],np.int32);
clf = tree.DecisionTreeClassifier()

def bdt():
	d_tree.binary_decision_tree(data,labels,'gini')

def skldt():
	clf.fit(data,labels)

print(timeit.timeit(bdt, number=10000))
print(timeit.timeit(skldt, number=10000))
# go()