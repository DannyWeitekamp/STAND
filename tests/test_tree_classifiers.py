import unittest
from numbaILP.tree_classifiers import *
from numba.typed import List, Dict
import numpy as np


# CRITERION = CRITERION
# SPLIT_CHOICE = SPLIT_CHOICE
# PRED_CHOICE = PRED_CHOICE

# _PRED_CHOICE_pure_majority = PRED_CHOICE_pure_majority

# def fit_decision_tree(x,y,missing_values=None):
# 	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
# 	out =fit_tree(x,y,
# 			missing_values=missing_values,
# 			criterion_enum=CRITERION_gini,
# 			split_enum=SPLIT_CHOICE_single_max,
# 			sep_nan=True
# 		 )
# 	return out

# def fit_ambiguity_tree(x,y,missing_values=None):
# 	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
# 	out =fit_tree(x,y,
# 			missing_values=missing_values,
# 			criterion_enum=CRITERION_gini,
# 			split_enum=SPLIT_CHOICE_all_max,
# 			cache_nodes=True,
# 		 )
# 	return out

# def predict_pm(tree,x):
# 	return predict_tree(tree,x,PRED_CHOICE_pure_majority,positive_class=1)

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




# def test_to_condtions():
# 	data2, labels2 = setup2()
# 	tree2 = fit_decision_tree(data2,labels2)
# 	treeA2 = fit_ambiguity_tree(data2,labels2)
# 	print_tree(treeA2)

# 	# compute_effective_purities(treeA2)
# 	conds = tree_to_conditions(tree2,1,only_pure_leaves=False)
# 	print(conds)
# 	conds = tree_to_conditions(treeA2,1,only_pure_leaves=True)
# 	print(conds)


	
if(__name__ == "__main__"):
	# test_simple()
	# test_bloop()
	test_basics1()
	test_basics2()
	test_basics3()
	test_as_conditions()
		


