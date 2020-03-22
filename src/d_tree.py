# cython: infer_types=True, language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import numba
from numba import types, njit,jitclass, guvectorize,vectorize,prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List
from numba.types import ListType, unicode_type
import timeit
from sklearn import tree
from compile_template import compile_template
from enum import IntEnum
# from enum import IntEnum
# class Entropy(IntEnum):
# 	gini = 0



		
	# if(criterion_func != None):
	# 	criterion_func(np.ones((1,2)))
	# 	return criterion_func
	# else:
	# 	return None

#uint[::1] counts
# totals = np.expand_dims(np.sum(counts,1)+1e-10,1)
# prob = counts / totals
# out = 1.0-(np.sum(np.square(prob),1))
@njit(f8[::1](u4[:,:]),nogil=True,fastmath=True,cache=True)
def gini(counts):
	out = np.empty(counts.shape[0], dtype=np.double)
	for j in range(counts.shape[0]):
		total = 0; #use epsilon? 1e-10
		for i in range(counts.shape[1]):
			total += counts[j,i];

		sum_sqr = 0.0;
		if(total > 0):
			for i in range(counts.shape[1]):
				prob = counts[j,i] / total;
				sum_sqr += prob * prob

		out[j] = 1.0-sum_sqr;
	return out

@njit(f8[::1](u4[:,:]),nogil=True,fastmath=True,cache=True)
def return_zero(counts):
	return np.zeros(counts.shape[0],dtype=np.double)

# @njit(nogil=True,fastmath=True,cache=True)
# def calc_criterion(counts,criterion_func):
# 	out = np.empty(counts.shape[0], dtype=np.double)
# 	out_v = out
	
# 	for i in range(counts.shape[0]):
# 		out_v[i] = criterion_func(counts[i])
# 	return out


@njit(nogil=True,fastmath=True,cache=True)
def counts_per_split(start_counts, x, y_inds):
	counts = np.zeros((x.shape[1],2,len(start_counts)),dtype=np.uint32);
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(x[i,j]):
				counts[j,1,y_inds[i]] += 1;	
	# counts[:,0,:] = start_counts - counts[:,1,:]
			else:
				counts[j,0,y_inds[i]] += 1;	
	return counts;




@njit(nogil=True,fastmath=True,cache=True)
def unique_counts(inp):
	counts = [];
	uniques = [];
	inds = np.zeros(len(inp),np.uint32);
	ind=0;
	last = 0;
	for i in range(1,len(inp)):
		if(inp[i-1] != inp[i]):
			counts.append(i-last);
			uniques.append(inp[i-1]);
			last = i;
			ind += 1;
		inds[i] = ind;
	counts.append((i+1)-last);
	uniques.append(inp[i]);

	c = np.asarray(counts,dtype=np.uint32)
	u = np.asarray(uniques,dtype=np.int32)
	return c, u, inds

# @njit(nogil=True,fastmath=True,cache=True)
# def ambiguity_tree(x, y):
#f8[::1](u4[:,:]))
# criterion_type = numba.deferred_type()
NUMBA_FUNCS = {
	"criterion" : {
		'gini' : gini,
		'giniimpurity' : gini,
		'zero' : return_zero
	},
	'split_chooser' : {
		'single' : None
	}	
}



	
	




# @njit(nogil=True,fastmath=True,cache=True)
def get_numba_func(ftype,name):
	name = name.lower().replace("_","")
	if(ftype in NUMBA_FUNCS):
		if(name in NUMBA_FUNCS[ftype]):
			return NUMBA_FUNCS[ftype][name]
		else:
			raise ValueError("Criterion %r not recognized." % name)
	else:
		raise ValueError("ftype %f not recognized" % ftype)




class ILPTree(object):
	def __init__(self,criterion='gini',split_chooser=''):
		self.criterion = criterion
		# self.nb_ilp_tree = NB_ILPTree()
		self.criterion_func = get_numba_func('criterion',self.criterion)

		# criterion_type.define(numba.typeof(gini))
	# def run_it(self,inp):
	# 	return self.nb_ilp_tree.run_it(inp,self.criterion_func)
	# def fit(self,x,y):
	# 	return self.nb_ilp_tree.fit(x,y,self.criterion_func)
class TreeTypes(IntEnum):
	NODE = 1
	LEAF = 2

# TN_deffered = deferred_type()

@jitclass([('ttype',	   i4),
		   ('index',	   i4),
		   ('split_on',optional(i4)),
		   ('left',    optional(i4)),
		   ('right',   optional(i4)),
		   ('counts',  optional(u4[:]))])
class TreeNode(object):
	def __init__(self):
		self.ttype = 0
		self.index = 0
		#For Nodes
		self.split_on = None
		self.left = None
		self.right = None
		#For Leaves
		self.counts = None

TN = TreeNode.class_type.instance_type

@jitclass([('nodes',ListType(TN))])
class Tree(object):
	def __init__(self):
		self.nodes = List.empty_list(TN)

TR = Tree.class_type.instance_type

# TN_deffered.define(TN)
# _tn = TreeNode()

@jitclass([('x', b1[:,:]),
		   ('y', u4[:]),
		   ('impurity', f8),
		   ('counts', u4[:]),
		   ('parent_node', TN)])
class SplitContext(object):
	def __init__(self,x,y,impurity,counts,parent_node):
		self.x = x
		self.y = y
		self.impurity = impurity
		self.counts = counts
		self.parent_node = parent_node

@njit(void(TN,i4,i4,i4),nogil=True,fastmath=True)
def assign_node(tn,split_on,left,right):
	tn.ttype = TreeTypes.NODE
	tn.split_on = split_on
	tn.left = left
	tn.right = right

@njit(void(TN,u4[:]),nogil=True,fastmath=True)
def assign_leaf(tn,counts):
	tn.ttype = TreeTypes.LEAF
	tn.counts = counts

@njit(TN(TR),nogil=True,fastmath=True)
def new_node(tree):
	tn = TreeNode()
	index = len(tree.nodes)
	tree.nodes.append(tn)
	tn.index = index
	return tn
# @njit(nogil=True,fastmath=True,cache=True)
# def new_Node(split_on,left=None,right=None):
# 	return TreeNode(TreeTypes.NODE,split_on,left,right,None)

# @njit(nogil=True,fastmath=True,cache=True)
# def new_Leaf(counts):
# 	return TreeNode(TreeTypes.LEAF,counts=counts)



@njit(nogil=True,fastmath=True,cache=True)
def test_fit(x,y):	
	out =fit_tree(x,y,
			criterion_func=gini
		 )
	return out

@njit(nogil=True,fastmath=True)
def fit_tree(x,y,criterion_func):
	# criterion_func = gini#get_criterion_func(criterion)
	sorted_inds = np.argsort(y)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);
	impurity = criterion_func(np.expand_dims(counts,0))[0]
	contexts = List()
	tree = Tree()
	contexts.append(SplitContext(x_sorted,y_inds,impurity,counts,new_node(tree)))

	# parent_node = TreeNode()

	while len(contexts) > 0:
		new_contexts = List()

		for i in range(len(contexts)):
			c = contexts[i]

			countsPS = counts_per_split(c.counts,c.x,c.y)
			flat_countsPS = countsPS.reshape((-1,countsPS.shape[2]))
			flat_impurities = criterion_func(flat_countsPS)
			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))

			#Sum of new impurities of left and right side of split
			total_split_impurity = impurities[:,0] + impurities[:,1];  
			impurity_decrease = c.impurity - (total_split_impurity);
			max_split = np.argmax(impurity_decrease);
			
			print(c.counts,impurity_decrease[max_split])

			if(impurity_decrease[max_split] <= 0):

				# pass
				# max_count = countsPS[max_split,0,:];
				print(c.counts,c.impurity)
				assign_leaf(c.parent_node, c.counts)
				# new_Leaf(countsPS[max_split,0,:])

			else:#if(total_split_impurity[max_split] > 0):
				node_l, node_r = new_node(tree), new_node(tree)
				# assign_node(c.parent_node,max_split,node_l,node_r)
				assign_node(c.parent_node,max_split,node_l.index,node_r.index)

				ms_impurity_l = impurities[max_split,0].item()
				ms_impurity_r = impurities[max_split,1].item()

				mask = c.x[:,max_split];
				print("HERE1")
				if(ms_impurity_l > 0):
					print("HERE1A",countsPS[max_split,0])
					sel = np.argwhere(np.logical_not(mask))[:,0]
					new_contexts.append(SplitContext(c.x[sel], c.y[sel],
						ms_impurity_l,countsPS[max_split,0], node_l))
				else:
					print("HERE1B")
					print(countsPS[max_split,0],ms_impurity_l)
					assign_leaf(node_l, countsPS[max_split,0])
				
				print("HERE2")
				if(ms_impurity_r > 0):
					print("HERE2A")
					sel = np.argwhere(mask)[:,0]
					new_contexts.append(SplitContext(c.x[sel], c.y[sel],
						ms_impurity_r,countsPS[max_split,1], node_r))
				else:
					print("HERE2B")
					print(countsPS[max_split,1],ms_impurity_r)
					assign_leaf(node_r, countsPS[max_split,1])
			# else:
			# 	assign_leaf(c.parent_node, c.counts)

		contexts = new_contexts
	for i in range(len(tree.nodes)):
		tn = tree.nodes[i]
		if(tn.ttype == TreeTypes.NODE):
			print(tn.index,": ",tn.split_on, tn.left, tn.right)
		else:
			print(tn.index,": ",tn.counts)
	return 0
# @jitclass([('criterion', unicode_type)])
# @jitclass([])
# class NB_ILPTree(object):
# 	def __init__(self):
# 		pass

# 	# @njit(nogil=True,fastmath=True,cache=True)
# 	def run_it(self,inp,criterion_func):
# 		# criterion_func = get_criterion_func(self.criterion)
# 		return criterion_func(inp)
# 	def fit():
		# pass





# @njit(nogil=True,fastmath=True,cache=True)
# def binary_decision_tree(x, y, criterion='gini', split_chooser=''):
# 	criterion_func = gini#get_criterion_func(criterion)
# 	sorted_inds = np.argsort(y)
# 	x_sorted = x[sorted_inds]
# 	y_sorted = y[sorted_inds]
# 	counts, u_ys, y_inds = unique_counts(y_sorted);
# 	impurity = criterion_func(np.expand_dims(counts,0))[0]
# 	contexts = List()
# 	contexts.append(SplitContext(x_sorted,y_inds,impurity,counts))

# 	while len(contexts) > 0:
# 		new_contexts = List()

# 		for i in range(len(contexts)):
# 			c = contexts[i]

# 			countsPS = counts_per_split(c.counts,c.x,c.y)
# 			flat_countsPS = countsPS.reshape((-1,countsPS.shape[2]))
# 			flat_impurities = criterion_func(flat_countsPS)
# 			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))

# 			#Sum of new impurities of left and right side of split
# 			total_split_impurity = impurities[:,0] + impurities[:,1];  
# 			impurity_decrease = impurity - (total_split_impurity);
# 			max_split = np.argmax(impurity_decrease);
			
# 			if(impurity_decrease[max_split] <= 0):
# 				max_count = countsPS[max_split,0,:];

# 			elif(total_split_impurity[max_split] > 0):
# 				ms_impurity_l = impurities[max_split,0].item()
# 				ms_impurity_r = impurities[max_split,1].item()

# 				mask = c.x[:,max_split];
# 				if(ms_impurity_l > 0):
# 					sel = np.argwhere(np.logical_not(mask))[:,0]
# 					new_contexts.append(SplitContext(c.x[sel], c.y[sel],
# 						ms_impurity_l,countsPS[max_split,0]))
# 				else:
# 					pass

# 				if(ms_impurity_r > 0):
# 					sel = np.argwhere(mask)[:,0]
# 					new_contexts.append(SplitContext(c.x[sel], c.y[sel],
# 						ms_impurity_r,countsPS[max_split,1]))
# 				else:
# 					pass
# 		contexts = new_contexts
# 	return 0

@njit(nogil=True,fastmath=True,cache=True)
def test1(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)

@njit(nogil=True,fastmath=True,cache=True)
def test2(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);

@njit(nogil=True,fastmath=True,cache=True)
def test3(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);
	cps = counts_per_split(counts,x_sorted,y_sorted)

@njit(nogil=True,fastmath=True,cache=True)
def test4(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);
	cps = counts_per_split(counts,x_sorted,y_sorted)
	mask = cps[:,2];
	sel = np.argwhere(np.logical_not(mask))[:,0]

@njit(nogil=True,fastmath=True,cache=True)
def test5(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);
	cps = counts_per_split(counts,x_sorted,y_sorted)
	mask = cps[:,2];
	sel = np.argwhere(np.logical_not(mask))[:,0]
	entropy = gini(counts)

@njit(nogil=True,fastmath=True,cache=True)
def test6(x,y):
	sorted_inds = np.argsort(y).astype(np.int32)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);
	cps = counts_per_split(counts,x_sorted,y_sorted)
	mask = cps[:,2];
	sel = np.argwhere(np.logical_not(mask))[:,0]
	entropy = gini(counts)
	context = SplitContext(x_sorted,y_sorted,entropy, counts)





if(__name__ == "__main__"):
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
	my_bdt = ILPTree()
	# nb_fit = my_bdt.nb_ilp_tree.fit

	def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

	def t1():
		test1(data,labels)

	def t2():
		test2(data,labels)

	def t3():
		test3(data,labels)

	def t4():
		test4(data,labels)

	def t5():
		test5(data,labels)

	def t6():
		test6(data,labels)

	def bdt():
		test_fit(data,labels)
		# binary_decision_tree(data,labels)
		# my_bdt.fit(data,labels)
		# nb_fit(data,labels,gini)

	def skldt():
		clf.fit(data,labels)

	def control():
		return 0
	
	N = 10000
	# f = get_criterion_func('gini')
	print(numba.typeof(gini))
	print(numba.typeof(unique_counts(labels)))

	# print(numba.typeof({}))
	# print("numba:", time_ms(bdt))
	# print("control:", time_ms(control))
	# print("t1:", time_ms(t1))
	# print("t2:", time_ms(t2))
	# print("t3:", time_ms(t3))
	# print("t4:", time_ms(t4))
	# print("t5:", time_ms(t5))
	# print("t6:", time_ms(t6))
	# print("sklearn:", time_ms(skldt))

	bdt()
	sorted_inds = np.argsort(labels)
	y_sorted = labels[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);

	# tree = ILPTree('zero')
	# print(tree.run_it(np.expand_dims(counts,0)))

	# tree = ILPTree('gini')
	# print(tree.run_it(np.expand_dims(counts,0)))
	# 

# cdef binary_decision_tree_Woop():
# 	split_tree(gini)
# def 
	# return 

# def binary_decision_tree(bool[:,:] x, ):
	

	# print(gini(x))






