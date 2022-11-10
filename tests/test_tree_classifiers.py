import unittest
from numbaILP.tree_classifiers import *
from numba.typed import List, Dict
import numpy as np
import re
import pytest

#### test_optimal_split ####

def test_optimal_split():
	'''Check that can find optimal splits on continous variables when they exist'''
	#
	miss_mask = np.empty((0,0),np.bool)
	N = 10
	xc = np.asarray([np.arange(N)],np.float64).T
	xb = np.zeros((0,0),np.bool)
	for i in range(N+1):
		y = np.concatenate([np.zeros(i,dtype=np.int64),np.ones(N-i,dtype=np.int64)])
		counts = np.array([i,N-i],dtype=np.int64)
		# xb, xc, y, miss_mask, base_impurity, counts, criterion_enum, total_enum, pos_ind, n_classes, sep_nan
		out = get_counts_impurities(xb, xc, y, miss_mask, 1.0, counts, CRITERION_gini, TOTAL_sum, 1, 2, True)
		countsPS, impurities, thresholds, ops = out
		# print(impurities)
		# assert (impurities == 0.0).all()
		# print(i, countsPS, impurities, thresholds)
		if(i ==0 or i == N): 
			#When the input is pure the threshold should be inf
			assert thresholds[0] == np.inf
			assert all(np.sum(countsPS[0],axis=1) == np.array([10,0]))
		else:
			assert thresholds[0] > i-1 and thresholds[0] < i
			assert all(np.sum(countsPS[0],axis=1) == np.array([i,N-i]))
	

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
	data1_flt = data1.astype(np.float64)

	print("------BASIC 1------")
	
	# dt = TreeClassifier('decision_tree')
	# dt.fit(data1,None,labels1) # Binary DT
	# print(dt)
	# print(dt.predict(data1,None))
	# assert np.sum(dt.predict(data1,None) == labels1) >= 6
	# dt.fit(None,data1_flt,labels1) # Continous DT 
	# print(dt)
	# print(dt.predict(None,data1_flt))
	# assert np.sum(dt.predict(None,data1_flt) == labels1) >= 6
	# dt.fit(data1, data1_flt, labels1) # Mixed DT 
	# print(dt)
	# print(dt.predict(data1,data1_flt))
	# assert np.sum(dt.predict(data1, data1_flt) == labels1) >= 6
	
	at = TreeClassifier('ambiguity_tree')
	at.fit(data1,None,labels1) # Binary AT
	print(at)
	# assert np.sum(at.predict(data1,None) == labels1) >= 7
	at.fit(None,data1_flt,labels1) # Continous AT
	print(at)
	print(at.predict(None,data1_flt))
	# assert np.sum(at.predict(None,data1_flt) == labels1) >= 7
	at.fit(data1,data1_flt,labels1) # Mixed AT
	print(at)
	print(at.predict(data1,data1_flt))
	# assert np.sum(at.predict(data1,data1_flt) == labels1) >= 7

	print("GREEDY COVER")
	gc = TreeClassifier('greedy_cover_tree')
	gc.fit(data1,None,labels1) # Binary AT
	print(gc)
	print(gc.predict(data1,None))
	assert np.sum(gc.predict(data1,None) == labels1) >= 6
	gc.fit(None,data1_flt,labels1) # Continous AT
	print(gc)
	print(gc.predict(None,data1_flt))
	assert np.sum(gc.predict(None,data1_flt) == labels1) >= 6

	#TODO: Mixed greedy_cover_tree causes segfault
	# gc.fit(data1,data1_flt,labels1) # Mixed AT
	# print(gc)
	# print(gc.predict(data1,data1_flt))
	# print("DONE")
	# assert np.sum(gc.predict(data1,data1_flt) == labels1) >= 6


def test_basics2():
	data2, labels2 = setup2()
	data2_flt = data2.astype(np.float64)

	dt = TreeClassifier('decision_tree')

	print("------BASIC 2------")
	
	# dt.fit(data2,None,labels2) # Binary DT
	# print(dt)
	# print(dt.predict(data2,None))
	# assert np.sum(dt.predict(data2,None) == labels2) >= 5
	# dt.fit(None,data2_flt,labels2) # Continous DT
	# print(dt)
	# print(dt.predict(None,data2_flt))
	# assert np.sum(dt.predict(None,data2_flt) == labels2) >= 5

	at = TreeClassifier('ambiguity_tree')
	at.fit(data2,None,labels2) # Binary AT
	print(at)
	print(at.predict(data2,None))
	assert np.sum(at.predict(data2,None) == labels2) >= 6
	at.fit(None,data2_flt,labels2) # Continous AT
	print(at)
	print(at.predict(None,data2_flt))
	assert np.sum(at.predict(None,data2_flt) == labels2) >= 6

	gc = TreeClassifier('greedy_cover_tree')
	gc.fit(data2,None,labels2) # Binary AT
	print(gc)
	print(gc.predict(data2,None))
	assert np.sum(gc.predict(data2,None) == labels2) >= 6
	gc.fit(None,data2_flt,labels2) # Continous AT
	print(gc)
	print(gc.predict(None,data2_flt))
	assert np.sum(gc.predict(None,data2_flt) == labels2) >= 6

def test_basics3():
	'''In this test it should pretty much fail to make a split regardless
		of implementation but it should still guess at 50% accuracy'''
	data3, labels3 = setup3()
	data3_flt = data3.astype(np.float64)

	print("------BASIC 3------")

	dt = TreeClassifier('decision_tree')
	dt.fit(data3,None,labels3) # Binary DT
	print(dt)
	print(dt.predict(data3,None))
	assert np.sum(dt.predict(data3,None) == labels3) >= 3
	dt.fit(None,data3_flt,labels3) # Continous DT
	print(dt)
	print(dt.predict(None,data3_flt))
	assert np.sum(dt.predict(None,data3_flt) == labels3) >= 3

	at = TreeClassifier('ambiguity_tree')
	at.fit(data3,None,labels3) # Binary AT
	print(at)
	print(at.predict(data3,None))
	assert np.sum(at.predict(data3,None) == labels3) >= 3
	at.fit(None,data3_flt,labels3) # Continous AT
	print(at)
	print(at.predict(None,data3_flt))
	assert np.sum(at.predict(None,data3_flt) == labels3) >= 3

	#Greedy cover should be able to figure this one out though
	gc = TreeClassifier('greedy_cover_tree')
	gc.fit(data3,None,labels3) # Binary AT
	print(gc)
	print(gc.predict(data3,None))
	assert np.sum(gc.predict(data3,None) == labels3) >= 4
	gc.fit(None,data3_flt,labels3) # Continous AT
	print(gc)
	print(gc.predict(None,data3_flt))
	assert np.sum(gc.predict(None,data3_flt) == labels3) >= 4

def setup_mixed():
	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,0], #1
	[0,0,1], #1
	[0,1,1], #1
	[1,1,1], #2
	[0,1,1], #2
	[1,1,1], #2
	],np.uint8);

	data_flt = np.asarray([
	[0,0,0],
	[0,0,0],
	[0,0,0],
	[0,0,1],
	[1,1,0],
	[0,1,0],
	],np.float64);

	
	labels = np.asarray([1,1,1,2,2,2],np.int64);

	return data, data_flt, labels

def setup_missing_mixed():
	data, data_flt, labels = setup_mixed()
	# missing_values = np.array([[0,0],[1,1],[0,2],[3,0],[0,5],[2,4],[3,3]],dtype=np.int64)

	bin_miss_mask = np.asarray([
	[1,0,1],
	[0,1,0],
	[0,0,0],
	[1,0,0],
	[0,0,0],
	[0,0,0],

	],np.bool)

	miss_mask = np.asarray([
	[0,0,1],
	[0,0,0],
	[0,1,0],
	[1,0,0],
	[0,0,0],
	[0,0,0],

	],np.bool)

	data[bin_miss_mask] = 255
	data_flt[miss_mask] = data_flt[miss_mask]+1
	
	return data, data_flt, miss_mask, labels 



def test_mixed():
	data, data_flt, labels = setup_mixed()

	dt = TreeClassifier('decision_tree')
	dt.fit(data,data_flt,labels) # Mixed DT
	assert np.sum(dt.predict(data,data_flt) == labels) >= 5

	at = TreeClassifier('ambiguity_tree')
	at.fit(data,data_flt,labels) # Mixed AT
	assert np.sum(at.predict(data,data_flt) == labels) >= 6


#### test_missing ####

def tree_is_pure(tree_classifier):
	'''Determine if a tree classifier has only pure leaves by parsing it's __str__  
	 representation'''
	pure = True
	for leaf_line in re.findall(r'LEAF.+',str(tree_classifier)):
		arr = [int(x) for x in re.findall(r'\d+',leaf_line.split(":")[1])]
		if (sum([x != 0 for x in arr]) != 1): pure = False
	return pure

def count_non_leaf_nodes(tree_classifier):
	'''Find the number of nodes in the tree'''
	return len(re.findall(r'NODE.+',str(tree_classifier)))
		

def setup_missing():
	data = np.asarray([
	[1,1], #1
	[1,1], #1
	[1,0], #1 <- mark second feature as missing
	[1,0], #2
	],np.bool);

	miss_mask = np.asarray([
	[0,0], #1
	[0,0], #1
	[0,1], #1 <- mark second feature as missing
	[0,0], #2
	],np.bool);


	labels = np.asarray([1,1,1,2],np.int64);
	return data, labels, miss_mask


def test_missing():
	data, labels, miss_mask = setup_missing()
	data_flt = data.astype(np.float64)

	dt = TreeClassifier('decision_tree')

	#It should not be possible to produce a pure tree with this data
	dt.fit(data,None,labels)
	print(dt)
	assert not tree_is_pure(dt)
	dt.fit(None,data_flt,labels)
	print(dt)
	assert not tree_is_pure(dt)
	dt.fit(data,data_flt,labels)
	print(dt)
	assert not tree_is_pure(dt)

	#However if the second feature of the third item happens to be a missing value
	#	then the tree should at least try to make a split at feature 1
	data[miss_mask] = 255

	dt.fit(data,None,labels)
	print(dt)
	assert count_non_leaf_nodes(dt) > 0
	dt.fit(None,data_flt,labels, miss_mask)
	print(dt)
	assert count_non_leaf_nodes(dt) > 0
	dt.fit(data,data_flt,labels, miss_mask)
	print(dt)
	assert count_non_leaf_nodes(dt) > 0

# def test_missing_ordering():
# 	N = 6
# 	# missing_values = np.empty((0,0),dtype=np.int64)#np.asarray([[1,0],[2,0],[3,0],[4,0]],np.int64)
# 	missing_values = np.asarray([[0,0], [1,0],[2,0],[3,0], [4,0]],np.int64)
# 	xc = np.asarray([[4,3,2,1,7,0,1]],np.float64).T
# 	xb = np.zeros((0,0),np.bool)
# 	y =  np.asarray([0,0,1,1,0,1,0],dtype=np.int64)
# 	counts = np.array([4,3],dtype=np.int64)

# 	out = get_counts_impurities(xb, xc, y, missing_values, 1.0, counts, CRITERION_gini, 2, True)
# 	countsPS, impurities, thresholds, ops = out
# 	print(thresholds)
# 	assert thresholds[0] == 0.5



def test_missing_mixed():
	data, data_flt, miss_mask, labels = setup_missing_mixed()
	print(data)
	print(data_flt)

	# It should not be possible to produce a pure tree with this data
	# which is basically basic2 but with some key features flipped
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(data, data_flt, labels) # Continous DT
	print(dt)
	assert not tree_is_pure(dt)
	nonleaf_w_o_missing = count_non_leaf_nodes(dt)

	# But if we mark the flipped features as missing then it should still work
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(data, data_flt, labels, miss_mask) # Continous DT
	print(dt)
	assert nonleaf_w_o_missing < count_non_leaf_nodes(dt)
	


#### test_nan #### 

def get_tree_op_counts(dt):
	d = {}
	for node in dt.tree.nodes:
		d[node.op_enum] = d.get(node.op_enum,0)+1
	return d

def setup_nan():
	n = np.nan
	data1 = np.asarray([
	[1,1,1], #1
	[1,1,1], #1
	[1,1,1], #1
	[1,1,n], #2
	[1,n,1], #2
	[n,1,1], #2
	],np.float64);

	data2 = np.asarray([
	[0,0,0], #0
	[0,0,0], #0
	[0,0,0], #0
	[0,0,n], #2
	[0,n,0], #2
	[n,0,0], #2
	],np.float64);

	data3 = np.asarray([
	[1,1,1], #1
	[1,1,1], #1
	[1,1,1], #1
	[1,1,n], #2
	[1,n,0], #2
	[n,1,0], #2
	],np.float64);


	data4 = np.asarray([
	[1,1,0], #1
	[1,1,0], #1
	[1,1,0], #1
	[1,1,n], #2
	[1,n,1], #2
	[n,1,1], #2
	],np.float64);

	miss_mask = np.asarray([
	[0,0,0], #1
	[0,0,0], #1
	[0,0,0], #1
	[0,0,1], #2
	[0,1,0], #2
	[1,0,0], #2
	],np.bool);

	labels = np.asarray([1,1,1,2,2,2],dtype=np.int64)

	return data1, data2, data3, data4, labels, miss_mask

def test_nan():
	data1, data2, data3, data4, labels, miss_mask = setup_nan()
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(None, data1, labels) 
	print(dt)
	assert tree_is_pure(dt)
	op_counts = get_tree_op_counts(dt)
	print(op_counts)
	assert op_counts.get(OP_ISNAN,0) > 0
	assert op_counts.get(OP_GE,0) == 0
	assert op_counts.get(OP_LT,0) == 0

	assert np.sum(dt.predict(None,data1) == labels) == 6

	#If the NaN values are missing we shouldn't be able to make any splits
	dt.fit(None, data1, labels, miss_mask) 
	assert not tree_is_pure(dt)

	dt.fit(None, data2, labels) 
	print(dt)
	assert tree_is_pure(dt)
	op_counts = get_tree_op_counts(dt)
	print(op_counts)
	assert op_counts.get(OP_ISNAN,0) > 0
	assert op_counts.get(OP_GE,0) == 0
	assert op_counts.get(OP_LT,0) == 0

	assert np.sum(dt.predict(None,data2) == labels) == 6

	#If the NaN values are missing we shouldn't be able to make any splits
	dt.fit(None, data2, labels, miss_mask) 
	assert not tree_is_pure(dt)

	dt.fit(None, data3, labels)
	print(dt)
	assert tree_is_pure(dt)
	op_counts = get_tree_op_counts(dt)
	assert op_counts.get(OP_ISNAN,0) == 0
	assert op_counts.get(OP_GE,0) == 1
	assert op_counts.get(OP_LT,0) == 0

	assert np.sum(dt.predict(None,data3) == labels) == 6

	dt.fit(None, data4, labels)
	print(dt)
	assert tree_is_pure(dt)
	op_counts = get_tree_op_counts(dt)
	assert op_counts.get(OP_ISNAN,0) == 0
	assert op_counts.get(OP_GE,0) == 0
	assert op_counts.get(OP_LT,0) == 1

	assert np.sum(dt.predict(None,data4) == labels) == 6



	
#### test_as_conditions ####

# def test_as_conditions():
# 	data2, labels2 = setup2()

# 	dt = TreeClassifier('decision_tree',positive_class=1)
# 	dt.fit(data2,None,labels2)

# 	at = TreeClassifier('ambiguity_tree',positive_class=1)
# 	at.fit(data2,None,labels2)

# 	conds = dt.as_conditions(only_pure_leaves=False)
# 	print(conds)

# 	conds = at.as_conditions(only_pure_leaves=True)
# 	print(conds)

#### BENCHMARKS ####

@pytest.mark.benchmark(group="fit_tree")
def test_b_bin_decision_tree_fit(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('decision_tree')

	def f():
		return dt.fit(data1,None, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)

@pytest.mark.benchmark(group="fit_tree")
def test_b_cont_decision_tree_fit(benchmark):
	data1, labels1 = setup1()
	data1_flt = data1.astype(np.float64)
	dt = TreeClassifier('decision_tree')

	def f():
		return dt.fit(None, data1_flt, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="fit_tree")
def test_b_bin_ambiguity_tree_fit(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('ambiguity_tree')

	def f():
		return dt.fit(data1, None, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="fit_tree")
def test_b_cont_ambiguity_tree_fit(benchmark):
	data1, labels1 = setup1()
	data1_flt = data1.astype(np.float64)
	dt = TreeClassifier('ambiguity_tree')

	def f():
		return dt.fit(None, data1_flt, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="fit_tree")
def test_b_sklearn_tree_fit(benchmark):
	data1, labels1 = setup1()
	clf = SKTree.DecisionTreeClassifier()

	def f():
		return clf.fit(data1, labels1)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)

#### Predict BENCHMARKS ####

@pytest.mark.benchmark(group="predict_tree")
def test_b_bin_decision_tree_predict(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('decision_tree')
	dt.fit(data1, None, labels1)

	def f():
		return dt.predict(data1,None)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="predict_tree")
def test_b_cont_decision_tree_predict(benchmark):
	data1, labels1 = setup1()
	data1_flt = data1.astype(np.float64)
	dt = TreeClassifier('decision_tree')
	dt.fit(None, data1_flt, labels1)

	def f():
		return dt.predict(None,data1_flt)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)

@pytest.mark.benchmark(group="predict_tree")
def test_b_bin_ambiguity_tree_predict(benchmark):
	data1, labels1 = setup1()
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(data1, None, labels1)

	def f():
		return dt.predict(data1,None)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="predict_tree")
def test_b_cont_ambiguity_tree_predict(benchmark):
	data1, labels1 = setup1()
	data1_flt = data1.astype(np.float64)
	dt = TreeClassifier('ambiguity_tree')
	dt.fit(None, data1_flt, labels1)

	def f():
		return dt.predict(None,data1_flt)

	benchmark.pedantic(f, warmup_rounds=1, iterations=100)


@pytest.mark.benchmark(group="predict_tree")
def test_b_sklearn_tree_predict(benchmark):
	data1, labels1 = setup1()
	clf = SKTree.DecisionTreeClassifier()
	clf.fit(data1, labels1)

	def f():
		clf.predict(data1)
		
	benchmark.pedantic(f, warmup_rounds=1, iterations=100)



if(__name__ == "__main__"):
	# test_optimal_split()
	test_basics1()
	test_basics2()
	test_basics3()
	# test_missing()
	
	# test_mixed()
	# test_missing_mixed()
	# test_nan()
	

	# test_missing_ordering()
	# test_as_conditions()
		


#Things that need to be tested
#   -missing values in both binary and continous
#   -Nan values 
