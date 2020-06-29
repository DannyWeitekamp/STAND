import unittest
from numbaILP.tree_classifiers import *
from numba.typed import List, Dict
import numpy as np


# CRITERION = CRITERION
# SPLIT_CHOICE = SPLIT_CHOICE
# PRED_CHOICE = PRED_CHOICE

# _PRED_CHOICE_pure_majority = PRED_CHOICE_pure_majority

def fit_decision_tree(x,y):
	out =fit_tree(x,y,
			criterion_enum=CRITERION_gini,
			split_enum=SPLIT_CHOICE_single_max,
			sep_nan=True
		 )
	return out

def fit_ambiguity_tree(x,y):
	out =fit_tree(x,y,
			criterion_enum=CRITERION_gini,
			split_enum=SPLIT_CHOICE_all_max,
			sep_nan=True
		 )
	return out

def predict_pm(tree,x):
	return predict_tree(tree,x,PRED_CHOICE_pure_majority,positive_class=1)

class TestTreeClassifiers(unittest.TestCase):
	def setUp(self):
		self.data1 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
		[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
		[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
		[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
		],np.bool);

		self.labels1 = np.asarray([3,1,1,1,2,2,2],np.int64);


		self.data2 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0,0,0,0,0], #1
		[0,0,1,0,0,0], #1
		[0,1,1,0,0,0], #1
		[1,1,1,0,0,1], #2
		[0,1,1,1,1,0], #2
		[1,1,1,0,1,0], #2
		],np.bool);

		self.labels2 = np.asarray([1,1,1,2,2,2],np.int64);
		self.data2 = self.data2[:,[1,0,2,3,4,5]]


		self.data3 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0], #1
		[1,0], #1
		[0,1], #1
		[1,1], #2
		],np.bool);

		self.labels3 = np.asarray([1,1,1,2],np.int64);

	def test_simple(self):
		tree1 = fit_decision_tree(self.data1,self.labels1)
		treeA1 = fit_ambiguity_tree(self.data1,self.labels1)

		self.assertGreaterEqual(np.sum(predict_pm(tree1,self.data1) == self.labels1),6)
		self.assertGreaterEqual(np.sum(predict_pm(treeA1,self.data1) == self.labels1),7)

		tree2 = fit_decision_tree(self.data2,self.labels2)
		treeA2 = fit_ambiguity_tree(self.data2,self.labels2)

		self.assertGreaterEqual(np.sum(predict_pm(tree2,self.data2) == self.labels2),5)
		self.assertGreaterEqual(np.sum(predict_pm(treeA2,self.data2) == self.labels2),6)

		tree3 = fit_decision_tree(self.data3,self.labels3)
		treeA3 = fit_ambiguity_tree(self.data3,self.labels3)

		self.assertGreaterEqual(np.sum(predict_pm(tree3,self.data3) == self.labels3),3)
		self.assertGreaterEqual(np.sum(predict_pm(treeA3,self.data3) == self.labels3),3)

		


