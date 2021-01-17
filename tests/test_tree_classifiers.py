import unittest
from numbaILP.tree_classifiers import *
from numba.typed import List, Dict
import numpy as np
import re

#### test_basics ####

def setup1():
	data1 = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
	],np.bool);

	labels1 = np.asarray([3,1,1,1,2,2,2],np.int64);
	return data1, labels1

def setup2():
	data2 = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,0,0,0,0], #1
	[0,0,1,0,0,0], #1
	[0,1,1,0,0,0], #1
	[1,1,1,0,0,1], #2
	[0,1,1,1,1,0], #2
	[1,1,1,0,1,0], #2
	],np.bool);

	labels2 = np.asarray([1,1,1,2,2,2],np.int64);
	data2 = data2[:,[1,0,2,3,4,5]]
	return data2, labels2

def setup3():
	data3 = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0], #1
	[1,0], #1
	[0,1], #1
	[1,1], #2
	],np.bool);

	labels3 = np.asarray([1,1,1,2],np.int64);
	return data3, labels3

def test_basics1():
	data1, labels1 = setup1()
	
	dt = TreeClassifier('decision_tree')
	dt.fit(data1,labels1)

	at = TreeClassifier('ambiguity_tree')
	at.fit(data1,labels1)

	assert np.sum(dt.predict(data1) == labels1) >= 6
	assert np.sum(at.predict(data1) == labels1) >= 7


def test_basics2():
	data2, labels2 = setup2()

	dt = TreeClassifier('decision_tree')
	dt.fit(data2,labels2)

	at = TreeClassifier('ambiguity_tree')
	at.fit(data2,labels2)

	assert np.sum(dt.predict(data2) == labels2) >= 5
	assert np.sum(at.predict(data2) == labels2) >= 6

def test_basics3():
	data3, labels3 = setup3()

	dt = TreeClassifier('decision_tree')
	dt.fit(data3,labels3)

	at = TreeClassifier('ambiguity_tree')
	at.fit(data3,labels3)

	assert np.sum(dt.predict(data3) == labels3) >= 3
	assert np.sum(at.predict(data3) == labels3) >= 3

#### test_missing ####

def tree_is_pure(tree_classifier):
	'''Determine if a tree classifier has only pure leaves by parsing it's __str__  
	 representation'''
	pure = True
	for leaf_line in re.findall(r'LEAF.+',str(tree_classifier)):
		arr = [int(x) for x in re.findall(r'\d+',leaf_line.split(":")[1])]
		if (sum([x != 0 for x in arr]) != 1): pure = False
	return pure

def setup_missing():
	data = np.asarray([
	[1,1], #1
	[1,1], #1
	[1,0], #1 <- mark second feature as missing
	[1,0], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int64);
	missing_values = np.asarray([[2,1]],np.int64)
	return data, labels, missing_values


def test_missing():
	data, labels, missing_values = setup_missing()

	dt = TreeClassifier('decision_tree')

	#It should not be possible to produce a pure tree with this data
	dt.fit(data,labels)
	assert not tree_is_pure(dt)

	#However if the second feature of the third item happened to me a missing value
	#	then it should be possible to produce two pure leaves
	dt.fit(data,labels, missing_values)
	assert tree_is_pure(dt)

	
#### test_as_conditions ####

def test_as_conditions():
	data2, labels2 = setup2()

	dt = TreeClassifier('decision_tree',positive_class=1)
	dt.fit(data2,labels2)

	at = TreeClassifier('ambiguity_tree',positive_class=1)
	at.fit(data2,labels2)

	conds = dt.as_conditions(only_pure_leaves=False)
	print(conds)

	conds = at.as_conditions(only_pure_leaves=True)
	print(conds)

#### BENCHMARKS ####

def test_b_decision_tree_fit(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('decision_tree')
	dt.fit(data1,labels1)
	fit = dt._fit
	@njit(cache=True)
	def f():
		return fit(data1, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


def test_b_ambiguity_tree_fit(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(data1,labels1)
	fit = dt._fit
	@njit(cache=True)
	def f():
		return fit(data1, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


def test_b_sklearn_tree_fit(benchmark):
	data1, labels1 = setup1()
	clf = SKTree.DecisionTreeClassifier()

	def f():
		return clf.fit(data1, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


	
if(__name__ == "__main__"):
	test_simple()
	test_bloop()
	test_basics1()
	test_basics2()
	test_basics3()
	test_missing()
	test_as_conditions()
		


