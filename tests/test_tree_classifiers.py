import unittest
from numbaILP.tree_classifiers import fit_tree, predict_tree, TreeClassifier, CRITERION, SPLIT_CHOICE
from numba.typed import List, Dict
import numpy as np


def fit_decision_tree(x,y):
	out =fit_tree(x,y,
			criterion_enum=CRITERION.gini,
			split_enum=SPLIT_CHOICE.single_max,
			sep_nan=True
		 )
	return out

def fit_ambiguity_tree(x,y):
	out =fit_tree(x,y,
			criterion_enum=CRITERION.gini,
			split_enum=SPLIT_CHOICE.all_max,
			sep_nan=True
		 )
	return out

def predict_pmg(tree,x):
	return predict_tree(tree,x,PRED_CHOICE.pure_majority_general,positive_class=1)

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

		print("PREDICT DT",predict_pmg(tree1,self.data1))
		print("PREDICT DT",predict_pmg(treeA1,self.data1))


		tree2 = fit_decision_tree(self.data2,self.labels2)
		treeA2 = fit_ambiguity_tree(self.data2,self.labels2)

		print("PREDICT DT",predict_pmg(tree2,self.data2))
		print("PREDICT DT",predict_pmg(treeA2,self.data2))


		tree3 = fit_decision_tree(self.data3,self.labels3)
		treeA3 = fit_ambiguity_tree(self.data3,self.labels3)

		print("PREDICT DT",predict_pmg(tree3,self.data3))
		print("PREDICT DT",predict_pmg(treeA3,self.data3))



