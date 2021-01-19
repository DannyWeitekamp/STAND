# cython: infer_types=True, language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import numba
from numba import types, njit, guvectorize,vectorize,prange, jit, literally
# from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType,ListType, unicode_type, NamedTuple,NamedUniTuple,Tuple
from collections import namedtuple
import timeit
from sklearn import tree as SKTree
from numbaILP.compile_template import compile_template
from enum import IntEnum
from numba.pycc import CC
from numbaILP.fnvhash import hasharray#, AKD#, akd_insert,akd_get
from operator import itemgetter
from numbaILP.structref import define_structref

@njit(nogil=True,fastmath=True,cache=False)
def unique_counts(inp):
	''' 
		Finds the unique classes in an input array of class labels
	'''
	counts = [];
	uniques = [];
	inds = np.zeros(len(inp),dtype=np.uint32);
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



#########  Impurity Functions #######
# class CRITERION(IntEnum):
# 	gini = 1
# 	return_zero = 2


@njit(f8[::1](u4[:,:]),nogil=True,fastmath=True,cache=True,inline='always')
def gini(counts):
	out = np.empty(counts.shape[0], dtype=np.double)
	for j in range(counts.shape[0]):
		total = 0; #use epsilon? 1e-10
		for i in range(counts.shape[1]):
			total += counts[j,i];

		s = 0.0;
		if(total > 0):
			for i in range(counts.shape[1]):
				prob = counts[j,i] / total;
				s += prob * (1.0-prob)

		out[j] = s;
	return out

@njit(f8[::1](u4[:,:]),nogil=True,fastmath=True,cache=True,inline='always')
def return_zero(counts):
	return np.zeros(counts.shape[0],dtype=np.double)


CRITERION_gini = 1
CRITERION_return_zero = 2


@njit(cache=True, inline='always')
def criterion_func(func_enum,counts):
	if(func_enum == 1):
		return gini(counts)
	elif(func_enum == 2):
		return return_zero(counts)
	return gini(counts)


######### Split Choosers ##########

# class SPLIT_CHOICE(IntEnum):
# 	single_max = 1
# 	all_max = 2

# @njit(i4[::1](f8[:]),nogil=True,fastmath=True,cache=True)
@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_single_max(impurity_decrease):
	'''A split chooser that expands greedily by max impurity 
		(i.e. this is the chooser for typical decision trees)'''
	return np.asarray([np.argmax(impurity_decrease)])

# @njit(i4[::1](f8[:]),nogil=True,fastmath=True,cache=True)
@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_all_max(impurity_decrease):
	'''A split chooser that expands every decision tree 
		(i.e. this chooser forces to build whole ambiguity tree)'''
	m = np.max(impurity_decrease)
	return np.where(impurity_decrease==m)[0]

SPLIT_CHOICE_single_max = 1
SPLIT_CHOICE_all_max = 2


@njit(cache=True,inline='always')
def split_chooser(func_enum,impurity_decrease):
	if(func_enum == 1):
		return choose_single_max(impurity_decrease)
	elif(func_enum == 2):
		return choose_all_max(impurity_decrease)
	return choose_single_max(impurity_decrease)

######### Prediction Choice Functions #########
# class PRED_CHOICE(IntEnum):
# 	majority = 1
# 	pure_majority = 2
# 	majority_general = 3
# 	pure_majority_general = 4



@njit(nogil=True,fastmath=True,cache=True,inline='always')
def get_pure_counts(leaf_counts):
	pure_counts = List()
	for count in leaf_counts:
		if(np.count_nonzero(count) == 1):
			pure_counts.append(count)
	return pure_counts

@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_majority(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority of leaves.''' 
	predictions = np.empty((len(leaf_counts),),dtype=np.int32)
	for i,count in enumerate(leaf_counts):
		predictions[i] = np.argmax(count)
	c,u, inds = unique_counts(predictions)
	_i = np.argmax(c)
	return u[_i]

@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_pure_majority(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority pure of leaves.'''
	pure_counts = get_pure_counts(leaf_counts)
	leaf_counts = pure_counts if len(pure_counts) > 0 else leaf_counts
	return choose_majority(leaf_counts,positive_class)

@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_majority_general(leaf_counts,positive_class):
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return 1
	return 0

@njit(nogil=True,fastmath=True,cache=True,inline='always')
def choose_pure_majority_general(leaf_counts,positive_class):	
	pure_counts = get_pure_counts(leaf_counts)
	leaf_counts = pure_counts if len(pure_counts) > 0 else leaf_counts
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return 1
	return 0


PRED_CHOICE_majority = 1
PRED_CHOICE_pure_majority = 2
PRED_CHOICE_majority_general = 3
PRED_CHOICE_pure_majority_general = 4

@njit(nogil=True,fastmath=True,cache=True,inline='always')
def pred_choice_func(func_enum,leaf_counts,positive_class):
	if(func_enum == 1):
		return choose_majority(leaf_counts,positive_class)
	elif(func_enum == 2):
		return choose_pure_majority(leaf_counts,positive_class)
	elif(func_enum == 3):
		return choose_majority_general(leaf_counts,positive_class)
	elif(func_enum == 4):
		return choose_pure_majority_general(leaf_counts,positive_class)
	return choose_majority(leaf_counts,positive_class)


# class TreeTypes(IntEnum):
# 	NODE = 1
# 	LEAF = 2


TreeTypes_NODE = 1
TreeTypes_LEAF = 2

######### Utility Functions for Fit/Predict  #########

@njit(nogil=True,fastmath=True,cache=True)
def counts_per_binary_split(x, y_inds, missing_values, n_classes, sep_nan=False):
	''' 
		Determines the number of elements of each class that would be in the resulting
		left, right and nan nodes if a split was made at each possible binary feature.
		Also outputs the index at which missing values stop being applicable to binary
		features.
	'''
	miss_i, miss_j = -1, -1
	if (len(missing_values) > 0):
		miss_i, miss_j = missing_values[0]
	miss_index = 1

	counts = np.zeros((x.shape[1], 2+sep_nan, n_classes),dtype=np.uint32);

	#Go through in Fortran order (Note: missing values should be ordered by j)
	for j in range(x.shape[1]):
		for i in range(x.shape[0]):
			if(i == miss_i and j == miss_j):
				# counts[j,0,y_inds[i]] += 1;	
				# counts[j,1,y_inds[i]] += 1;	
				if(miss_index < len(missing_values)):
					miss_i, miss_j = missing_values[miss_index]
					miss_index += 1
				else:
					miss_i, miss_j = -1, -1
			elif(sep_nan and np.isnan(x[i,j])):
				counts[j,2,y_inds[i]] += 1;	
			else:
				if(x[i,j]):
					counts[j,1,y_inds[i]] += 1;	
				else:
					counts[j,0,y_inds[i]] += 1;	
	return counts, miss_index






@njit(nogil=True,fastmath=True,cache=True)
def r_l_n_split(x, missing, sep_nan=False):
	'''Similar to argwhere applied 3 times each for 0,1 and nan, but does all
		three at once.'''
	nl,nr,nn = 0,0,0
	l = np.empty(x.shape,np.uint32)
	r = np.empty(x.shape,np.uint32)
	n = np.empty(x.shape,np.uint32)

	next_missing = missing[0] if len(missing) > 0 else -1
	m_ind = 1
	
	for i in range(len(x)):
		x_i = x[i]

		if(i == next_missing):
			# n[nn] = i
			# nn += 1
			# r[nr] = i
			# nr += 1
			# l[nl] = i
			# nl += 1

			next_missing = missing[m_ind]
			m_ind += 1
		elif(sep_nan and x_i == 255):
			n[nn] = i
			nn += 1
		else:
			if(x[i]):
				r[nr] = i
				nr += 1
			else:
				l[nl] = i
				nl += 1
	return l[:nl], r[:nr], n[:nn]


###### Array Keyed Dictionaries ######

BE = Tuple([u1[::1],i4])
BE_List = ListType(BE)

@njit(nogil=True,fastmath=True)
def akd_insert(akd,_arr,item,h=None):
	'''Inserts an i4 item into the dictionary keyed by an array _arr'''
	arr = _arr.view(np.uint8)
	if(h is None): h = hasharray(arr)
	elems = akd.get(h,List.empty_list(BE))
	is_in = False
	for elem in elems:
		if(len(elem[0]) == len(arr) and
			(elem[0] == arr).all()): 
			is_in = True
			break
	if(not is_in):
		elems.append((arr,item))
		akd[h] = elems

@njit(nogil=True,fastmath=True)
def akd_get(akd,_arr,h=None):
	'''Gets an i4 from a dictionary keyed by an array _arr'''
	arr = _arr.view(np.uint8)
	if(h is None): h = hasharray(arr) 
	if(h in akd):
		for elem in akd[h]:
			if(len(elem[0]) == len(arr) and
				(elem[0] == arr).all()): 
				return elem[1]
	return -1

'''
TreeNode: A particular node in the tree
	ttype -- Indicates if it is a leaf or node
	index -- The location of the node in the list of all nodes
	split_on -- If is a non-leaf node, the set of splits made on this node
		can be more than one in the case of ambiguity tree
	left -- For each split in 'split_on' the index of the node to the left
	right -- For each split in 'split_on' the index of the node to the right
	nan -- For each split in split_on the index of the node in the nan slot
	counts -- If is a leaf node the number of samples of each class falling in it
'''
# TreeNode = namedtuple("TreeNode",['ttype','index','split_data','counts'])
# TN = NamedTuple([i4,i4,ListType(i4[:]),u4[::1]],TreeNode)

treenode_fields = [
	('ttype',i4),
	('index',i4),
	('split_data',ListType(i4[:])),
	('counts', u4[::1])
]

TreeNode, TN = define_structref("TreeNode",treenode_fields)			

'''
SplitContext: An object holding relevant local variables of the tree after a split.
	This struct is used to avoid using recursion.
	inds -- A list of indicies of samples which fall in the present branch of the tree.
	impurity -- The impurity of this branch of the tree.
	counts -- The number of samples of each class.
	parent node -- The node from which this branch was produced.
'''

SplitContext = namedtuple("SplitContext",['inds','impurity','counts','parent_node'])
SC = NamedTuple([u4[::1],f8,u4[::1],i4],SplitContext)

i4_arr = i4[:]



Tree, TreeType = define_structref("Tree",[("nodes",ListType(TN)),('u_ys', i4[::1])])			


######### Fit #########

#NOTE: new_node is probably commented out in fit_tree and replaced by an inline implementation
#	numba's inlining isn't quite mature enough to not take a slight performance hit.
@njit(cache=True,locals={"NODE":i4,"LEAF":i4,'node':i4},inline='always')
def new_node(locs,split,new_inds, impurities,countsPS,ind):
	node_dict,nodes,new_contexts,cache_nodes = locs
	NODE, LEAF = i4(1), i4(2) #np.array(1,dtype=np.int32).item(), np.array(2,dtype=np.int32).item()
	node = i4(-1)
	if (cache_nodes): node= akd_get(node_dict,new_inds)
	if(node == -1):
		node = i4(len(nodes))
		if(cache_nodes): akd_insert(node_dict,new_inds,node)
		ms_impurity = impurities[split,ind].item()
		if(ms_impurity > 0.0):
			nodes.append(TreeNode(NODE,node,List.empty_list(i4_arr),countsPS[split,ind]))
			new_contexts.append(SplitContext(new_inds,
				ms_impurity,countsPS[split,ind], node))
		else:
			nodes.append(TreeNode(LEAF,node,List.empty_list(i4_arr),countsPS[split,ind]))
	return node

# @njit(cache=True,locals={"NODE":i4,"LEAF":i4,'node':i4},inline='always')
# def new_node(node_dict, nodes, new_contexts, cache_nodes, split, new_inds, impurities,countsPS,ind):
# 	NODE, LEAF = 1, 2
# 	node_id = -1
# 	if (cache_nodes): node_id = akd_get(node_dict,new_inds_n)
# 	if(node_id == -1):
# 		node_n = len(nodes)
# 		if(cache_nodes): akd_insert(node_dict,new_inds_n,node_n)
# 		ms_impurity_n = impurities[split,2].item()
# 		if(ms_impurity_n > 0):
# 			nodes.append(TreeNode(NODE,node_n,List.empty_list(i4_arr),countsPS[split,2]))
# 			new_contexts.append(SplitContext(new_inds_n,
# 				ms_impurity_n,countsPS[split,2], node_n))
# 		else:
# 			nodes.append(TreeNode(LEAF,node_n,List.empty_list(i4_arr),countsPS[split,2]))
# 	return node_id

@njit(cache=True)
def get_counts_impurities(xb, xc, y, missing_values, base_impurity, criterion_enum, n_classes, sep_nan):
	#NOTE: This function assumes that the elements [i,j] of missing_values is sorted by 'j'  

	n_b, n_c = xb.shape[1], xc.shape[1]
	countsPS = np.empty((n_b+n_c, 2+sep_nan, n_classes),dtype=np.uint32)
	impurities = np.empty((n_b+n_c, 2+sep_nan),dtype=np.float64)

	#Handle binary case
	countsPS_n_b, miss_index = counts_per_binary_split(xb, y, missing_values, n_classes, sep_nan)
	countsPS[:n_b] = countsPS_n_b
	flat_impurities = criterion_func(criterion_enum, countsPS_n_b.reshape((-1,n_classes)))
	impurities[:n_b] = flat_impurities.reshape((n_b, countsPS.shape[1]))


	#Handle continous case	
	# missing_values = missing_values[miss_index:]
	thresholds = np.empty((n_c,),dtype=np.float64)
	for j in range(n_c):
		#Sort on feature j so we can decide where the threshold will go
		xc_j = xc[:,j]
		srt_inds = np.argsort(xc_j)
		xc_j = xc_j[srt_inds]
		y_j = y[srt_inds]
		# print()
		# print("xc_j",xc_j)
		# print("y_j",y_j)

		#Numpy puts nan's at the end of a sort, find where they start 
		nan_start = len(xc_j)
		for i in range(len(xc_j)-1,-1,-1):
			if(not np.isnan(xc_j[i])): break
			nan_start = i
		
		# Find left(0) and right(1) cumulative counts for splitting at each possible threshold
		#  i.e figure out what child counts looks like if we put the threshold inbetween
		#  each of the sorted feature pairs.
		#  Also find and interval [pure_min,pure_max) such that all splits outside of it
		#  have a pure right or left (e.g. y_j=[0,0,1,0,1,1,1] has pure_min=2,pure_max=4)
		cum_counts = np.zeros((nan_start+1, 2, n_classes),dtype=np.int32)
		# pure_min, pure_target = 0, y_j[0]
		# print(len(missing_values), miss_index)
		miss_start = miss_index
		miss_i, miss_j = -1, -1
		if (miss_start < len(missing_values)):
			miss_i, miss_j = missing_values[miss_start]	
		for i in range(nan_start):
			y_ij = y_j[i]
			cum_counts[i+1, 0] = cum_counts[i, 0]
			if(miss_i == i and miss_j == j):
				#Ignore this feature since it's missing
				if(miss_index < len(missing_values)):
					miss_i, miss_j = missing_values[miss_index]
					miss_index += 1
				else:
					miss_i, miss_j = -1, -1
			else:
				cum_counts[i+1, 0, y_ij] += 1
				# if(y_ij == pure_target):
				# 	pure_min = i+1
				# else:
				# 	pure_target = -1

		# pure_max, pure_target = nan_start, y_j[-1]
		miss_index = miss_start
		miss_i, miss_j = -1, -1
		if (miss_start < len(missing_values)):
			miss_i, miss_j = missing_values[miss_start]	
		for i in range(nan_start,0,-1):
			y_ij = y_j[i-1]
			cum_counts[i-1, 1] = cum_counts[i, 1]
			if(miss_i == i and miss_j == j):
				#Ignore this feature since it's missing
				if(miss_index < len(missing_values)):
					miss_i, miss_j = missing_values[miss_index]
					miss_index += 1
				else:
					miss_i, miss_j = -1, -1
			else:
				cum_counts[i-1, 1, y_ij] += 1
				# if(y_ij == pure_target):
				# 	pure_max = i
				# else:
				# 	pure_target = -1
		# print(cum)
		# print("interval",pure_min, pure_max)
		# if(j == 0): base_impurity = criterion_func(criterion_enum, cum_counts[0])
			
		#Find all 'i' s.t. xc_j[i] != xc_j[i+1] (i.e. candidate pairs for threshold)
		#  only consider indicies inside the pure interval
		thresh_inds, c = np.empty((nan_start,),dtype=np.int32), 0
		for i in range(1, nan_start+1):
			if(xc_j[i] != xc_j[i-1]):
				thresh_inds[c] = i
				c += 1

		
		#If every value is the same then just use i=0
		 # if (c > 0) else np.zeros((1,),dtype=np.int32)
		best_impurity = np.zeros((2,),dtype=np.float64)
		best_impurity[0] = base_impurity
		if(c > 0):
			thresh_inds = thresh_inds[:c]

			# print("thresh_inds",thresh_inds)
			# print(cum_counts)
			
			best_total_impurity, best_ind = np.inf, -1
			# print("thresh_inds", thresh_inds)
			for t_i in thresh_inds:
				impurity = criterion_func(criterion_enum, cum_counts[t_i])
				total_impurity = np.sum(impurity)
				# print(t_i, total_impurity, best_total_impurity)
				if(total_impurity < best_total_impurity):
					best_impurity, best_total_impurity, best_ind = impurity, total_impurity, t_i
			thresh = (xc_j[best_ind-1] + xc_j[best_ind]) / 2.0 #if best_ind != 0 else np.inf
		else:
			# best_impurity, best_ind = np.zeros((2,),dtype=np.float64), nan_start
			thresh = np.inf

		# print(best_ind,thresh)
		countsPS[n_b+j,:2] = cum_counts[best_ind]
		impurities[n_b+j,:2] = best_impurity
		thresholds[j] = thresh

		if(sep_nan):
			nan_counts = np.empty((1,cum_counts.shape[-1]),dtype=np.int64)
			nan_counts[0] = 0
			for i in range(nan_start,len(y_j)):
				nan_counts[0,y_j[i]] += 1

			countsPS[n_b+j,2] = nan_counts
			impurities[n_b+j,2] = criterion_func(criterion_enum, nan_counts)[0]

			# 	print("I",i)
			# print(nan_counts)


		# else:
		# 	thresh = xc_j[0]
		# print("OUT", j, best_ind)
		# print(countsPS[n_b+j])
		# print(thresholds[j], total_split_impurities[n_b+j], )
	return countsPS, impurities, thresholds










@njit(cache=True, locals={"ZERO":i4,"NODE":i4,"LEAF":i4,"n_nodes":i4,"node_l":i4,"node_r":i4,"node_n":i4,"split":i4})
def fit_tree(x_bin, x_cont, y, missing_values, criterion_enum, split_enum, sep_nan=False, cache_nodes=False):
	'''Fits a decision/ambiguity tree'''

	#ENUMS definitions necessary if want to use 32bit integers since literals default to 64bit
	ZERO, NODE, LEAF = 0, 1, 2

	sorted_inds = np.argsort(y)
	x_bin_sorted = x_bin[sorted_inds]
	x_cont_sorted = x_cont[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);
	n_classes = len(u_ys)
	impurity = criterion_func(criterion_enum,np.expand_dims(counts,0))[0]

	contexts = List.empty_list(SC)
	contexts.append(SplitContext(np.arange(0,len(y),dtype=np.uint32),impurity,counts,ZERO))

	node_dict = Dict.empty(u4,BE_List)
	nodes = List.empty_list(TN)
	nodes.append(TreeNode(NODE,ZERO,List.empty_list(i4_arr),counts))
	

	while len(contexts) > 0:
		new_contexts = List.empty_list(SC)
		# locs = (node_dict,nodes,new_contexts,cache_nodes)
		for i in range(len(contexts)):
			c = contexts[i]
			c_xb, c_xc, c_y = x_bin_sorted[c.inds], x_cont_sorted[c.inds], y_inds[c.inds]

			# c_xb, c_y = x_sorted[c.inds], y_inds[c.inds]

			# countsPS = counts_per_split(c_x, c_y, n_classes, missing_values, sep_nan)
			# # print("M PS:", missing_values, "\n", countsPS)
			# flat_impurities = criterion_func(criterion_enum,countsPS.reshape((-1,countsPS.shape[2])))
			# impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))

			countsPS, impurities, thresholds =  \
				get_counts_impurities(c_xb, c_xc, c_y, missing_values, c.impurity,
										criterion_enum, n_classes, sep_nan)
			# print("IMP:", impurities)

			#Sum of new impurities of left and right side of split
			total_split_impurity = impurities[:,0] + impurities[:,1];
			if(sep_nan): total_split_impurity += impurities[:,2]
			impurity_decrease = c.impurity - (total_split_impurity);
			# print(total_split_impurity)
			# print(impurity_decrease)
			splits = split_chooser(split_enum, impurity_decrease)


			for j in range(len(splits)):
				split = splits[j]

				if(impurity_decrease[split] <= 0.0):
					nodes[c.parent_node]=TreeNode(LEAF,c.parent_node,List.empty_list(i4_arr),c.counts)
				else:
					if (split < c_xb.shape[1]):
						mask = c_xb[:,split]
					else: 
						mask = c_xc[:,split-c_xb.shape[1]] >= thresholds[split-c_xb.shape[1]]
					missing = np.argwhere(missing_values[:,1] == split)[:,0]
					# print("missing", split, missing)
					node_l, node_r, node_n = -1, -1, -1
					new_inds_l, new_inds_r, new_inds_n = r_l_n_split(mask,missing)
					locs = (node_dict, nodes,new_contexts, cache_nodes)
					
					#New node for left.
					node_l = new_node(locs,split,new_inds_l, impurities,countsPS, literally(0))

					#New node for right.
					node_r = new_node(locs,split,new_inds_r, impurities,countsPS,literally(1))

					#New node for NaN values.
					if(sep_nan and len(new_inds_n) > 0):
						node_n = new_node(locs,split,new_inds_n, impurities,countsPS,literally(2))
						
					#If is continous bitcast threshold to an i4 else set to 1 i.e. 1e-45
					thresh = np.float32(thresholds[split-c_xb.shape[1]]).view(np.int32) if split >= c_xb.shape[1] else 1
					split_data = np.array([split, thresh, node_l, node_r, node_n],dtype=np.int32)
					nodes[c.parent_node].split_data.append(split_data)

		contexts = new_contexts

	out = Tree(nodes,u_ys)
	# out = encode_tree(nodes,u_ys)
	return out

split_dtype = np.dtype([('split', np.int32), ('thresh', np.float32), ('node_l', np.int32), ('node_r', np.int32), ('node_n', np.int32)])

@njit(nogil=True,fastmath=True)
def encode_tree(nodes,u_ys):
	'''Takes a list of nodes and encodes them into a 1d-int32 numpy array. 
		The encoding is [length-node_parts, *[nodes...], class_ids] with:
			for each node in nodes : [len_encoding, ttype, index,*[splits...],*counts[:]]
			for each split in split : [feature_index, thresh, offset_left, offset_right, offset_nan]
		Note: Nodes with ttype=LEAF have no splits, only counts
		Note: This is done because (at least at numba 0.50.1) there is a significant perfomance 
		cost associate with unboxing Lists of NamedTuples, this seems to not be the case if
		the list is contained inside a jitclass, but jitclasses are not cacheable or AOT compilable
	'''
	n_classes = len(nodes[0].counts)
	out_node_slices = np.empty((len(nodes)+1,),dtype=np.int32)
	
	offset = 1 
	out_node_slices[0] = offset
	for i,node in enumerate(nodes):
		l = 4 + len(node.split_data)*5 + n_classes
		offset += l 
		out_node_slices[i+1] = offset
	out = np.empty((offset+len(u_ys)),dtype=np.int32)
	out[0] = np.array(offset,dtype=np.int32).item()
	for i,node in enumerate(nodes):
		ind = out_node_slices[i]

		out[ind+0] = out_node_slices[i+1]-out_node_slices[i]
		out[ind+1] = node.ttype 
		out[ind+2] = node.index
		out[ind+3] = len(node.split_data)
		ind += 4
		for sd in node.split_data:
			out[ind+0] = sd[0]; 
			out[ind+1] = sd[1];
			out[ind+2] = out_node_slices[sd[2]] if sd[2] != -1 else -1; 
			out[ind+3] = out_node_slices[sd[3]] if sd[2] != -1 else -1; 
			out[ind+4] = out_node_slices[sd[4]] if sd[3] != -1 else -1; 
			ind += 5
		out[ind:out_node_slices[i+1]] = node.counts
	out[out_node_slices[-1]:] = u_ys.astype(np.int32)

	return out



######### Predict #########
@njit(cache=True,inline='always')
def _unpack_node(tree,node_offset):
	'''Takes a tree encoded with encode_tree and the offset where a nodes is located in it
		and returns the ttype, index, splits, counts of that node. '''
	l  = tree[node_offset]
	slc = tree[node_offset:node_offset+l]
	ttype = slc[1]
	index = slc[2]
	if(ttype == TreeTypes_NODE):
		splits = slc[4:4+slc[3]*5].reshape(slc[3],5)
	else:
		splits = None
	counts = slc[4+slc[3]*5:]
	
	return ttype, index, splits, counts

@njit(cache=True,inline='always')
def _indexOf(tree,node_offset):
	'''Takes a tree encoded with encode_tree and the offset where a nodes is and returns
	   just the index of the node.'''
	return tree[node_offset+2]


@njit(cache=True,inline='always')
def _get_y_order(tree):
	'''Takes a tree encoded with encode_tree and the offset where a nodes is and returns
	   just the index of the node.'''
	return tree[tree[0]:]

condition_dtype = np.dtype([('feature', np.int32),#])
		                     ('nominal', np.uint8),
		                     ('pos_or_gt', np.uint8),
		                     ('thresh', np.float32)])

@njit(cache=True)
def _new_cond(feature,nominal,pos_or_gt,thresh):
	c = np.empty((1,),dtype=condition_dtype)
	c[0].feature = feature
	c[0].nominal = nominal
	c[0].pos_or_gt = pos_or_gt
	c[0].thresh = thresh
	return c


@njit(cache=True)
def _remove_over_constrained(conds):
	over_constrained_pairs = List()
	for i in range(len(conds)):
		cond = conds[i]
		for j in range(i-1,-1,-1):
			if( len(conds[j]) == len(cond) and
				((conds[j].feature == cond.feature) &
				 (conds[j].nominal == cond.nominal) &
				 (conds[j].thresh == cond.thresh)).all()
				):
				diff_conditions = (conds[j].pos_or_gt != cond.pos_or_gt)
				if(np.sum(diff_conditions) == 1):
					loc = np.argmax(diff_conditions)
					over_constrained_pairs.append((i,j,loc))
	replaced = Dict()
	out = List()
	for i,j,loc in over_constrained_pairs:
		replaced[i] = 1; replaced[j] = 1;
		out.append(np.delete(conds[i],loc))
	for i in range(len(conds)):
		if(i not in replaced):
			out.append(conds[i])
	return out



@njit(cache=True)
def _remove_duplicates(conds):
	new_conds = List()
	for i in range(len(conds)):
		cond = conds[i]
		is_dup = False
		for j in range(i-1,-1,-1):
			if( len(conds[j]) == len(cond) and
				((conds[j].feature == cond.feature) &
				 (conds[j].nominal == cond.nominal) &
			 	 (conds[j].pos_or_gt == cond.pos_or_gt) &
				 (conds[j].thresh == cond.thresh)).all()
				):
				is_dup = True
		if(not is_dup): new_conds.append(cond)
	return new_conds

purity_count = np.dtype([('parent', np.int32),#])
	                     ('is_pure', np.uint8)])

NodePurity = namedtuple("NodePurity",['parents','is_pure'])
NP = NamedTuple([ListType(i4),u1],NodePurity)

# @njit(cache=True)
# def _new_purity_count(parent,is_pure):
# 	c = np.empty((1,),dtype=condition_dtype)
# 	c[0].parent = parent
# 	c[0].is_pure = is_pure
# 	return c


@njit(cache=True,locals={"ZERO":i4,"ONE":i4,"UND":u1, "is_pure":u1})
def compute_effective_purities(tree):
	ZERO, ONE, UND = 0, 1,-1
	nodes = List.empty_list(i4); nodes.append(ONE)
	purities = Dict.empty(i4,NP)
	purities[ONE] = NodePurity(List.empty_list(i4),UND)
	to_resolve = List()
	while len(nodes) > 0:
		new_nodes = List()
		for node in nodes:
			print("NODE", node)
			ttype, index, splits, counts = _unpack_node(tree,node)
			if(ttype == TreeTypes_NODE):
				for j,s in enumerate(splits):
					split_on, thresh, left, right, nan  = s[0],s[1],s[2],s[3],s[4]
					if(left != -1): 
						new_nodes.append(left)
						# l_i = _indexOf(tree,left)
						l_purity = purities[left] = purities.get(left,NodePurity(List.empty_list(i4),UND))
						l_purity.parents.append(node)
					if(right != -1):
						new_nodes.append(right)
						# r_i = _indexOf(tree,right)
						r_purity = purities[right] = purities.get(right,NodePurity(List.empty_list(i4),UND))
						r_purity.parents.append(node)
					if(nan != -1):
						new_nodes.append(nan)
						# n_i = _indexOf(tree,nan)
						n_purity = purities[nan] = purities.get(nan,NodePurity(List.empty_list(i4),UND))
						n_purity.parents.append(node)
					
			else:
				is_pure = (np.count_nonzero(counts) == 1)
				leaf_purity = purities[node]
				purities[node] = NodePurity(leaf_purity.parents,is_pure)
				to_resolve.append(node)
		nodes = new_nodes
	# print("MID")				
	is_leaf_level = True
	while len(to_resolve) > 0:
		new_to_resolve = List()
		for node in to_resolve:
			purity = purities[node]
			for parent in purity.parents:
				if(parent == -1): continue
				# print("PAR",parent)
				parent_purity = purities[parent]
				if(parent_purity.is_pure == UND or (is_leaf_level and parent_purity.is_pure == 1)):
					purities[parent] = NodePurity(parent_purity.parents,purity.is_pure)
					# print("P",parent,purity.is_pure)
				new_to_resolve.append(parent)
		is_leaf_level = False
		to_resolve = new_to_resolve
	for _p, purity in purities.items():
		print("ISPURE:",_indexOf(tree,_p),purity.is_pure)

						
					# pur
					# parents = purities[index].parents
		

		

@njit(nogil=True,cache=True,locals={"ONE":i4,"is_nom":u1,"POS":u1,"NEG":u1,"NAN":u1,"FZERO":f4})
def tree_to_conditions(tree,target_class,only_pure_leaves=False):
	ONE = 1 
	POS,NEG,NAN = 1,0,-1
	FZERO = 0.0
	y_uvs = tree.u_ys#_get_y_order(tree)
	target = -1
	for i,y_uv in enumerate(y_uvs):
		if(y_uv == target_class): target = i; break;
	if(target == -1): raise ValueError("target_class not found in tree.")
	print('target',target)
	# assert target != -1, ("Tree does not contain class " + str(positive_class))
	nodes = List.empty_list(i4); nodes.append(ONE)
	conds = List(); conds.append(np.empty((0,),dtype=condition_dtype))
	leafs = List()
	out_conds = List()
	while len(nodes) > 0:
		new_nodes = List()
		new_conds = List()
		for cond,node in zip(conds,nodes):
			print('cond',cond)
			ttype, index, splits, counts = _unpack_node(tree,node)
			if(ttype == TreeTypes_NODE):
				is_nom = 1
				for j,s in enumerate(splits):
					split_on, left, right, nan  = s[0],s[1],s[2],s[3]
					if(only_pure_leaves):
						l_ttype, _, _, l_counts = _unpack_node(tree,left)
						r_ttype, _, _, r_counts = _unpack_node(tree,right)
						n_ttype, _, _, n_counts = _unpack_node(tree,nan)
						if(l_ttype == TreeTypes_LEAF and (np.count_nonzero(l_counts) != 1)): continue
						if(r_ttype == TreeTypes_LEAF and (np.count_nonzero(r_counts) != 1)): continue
						if(n_ttype == TreeTypes_LEAF and (np.count_nonzero(n_counts) != 1)): continue
					if(left != -1): 
						new_nodes.append(left)
						new_conds.append(np.append(cond,_new_cond(split_on,is_nom,NEG,FZERO)))
					if(right != -1):
					 	new_nodes.append(right)
					 	new_conds.append(np.append(cond,_new_cond(split_on,is_nom,POS,FZERO)))
					if(nan != -1):
						new_nodes.append(nan)
						new_conds.append(np.append(cond,_new_cond(split_on,is_nom,NAN,FZERO)))
			else:

				# is_target_leaf = np.argmax(counts) == target
				# print(index, counts,is_target_leaf,(not only_pure_leaves or (np.count_nonzero(counts) == 1)))
				#if(is_target_leaf):# and (not only_pure_leaves or (np.count_nonzero(counts) == 1))):
				if(np.argmax(counts) == target):
					leafs.append(counts)
					out_conds.append(cond)
					
		nodes = new_nodes
		conds = new_conds
	conds = out_conds
	out = np.empty((len(conds),))
	for i in range(len(conds)):
		sort_inds = np.argsort(conds[i].feature) 
		conds[i] = conds[i][sort_inds]
		print(conds[i])
	conds = _remove_over_constrained(conds)
	conds = _remove_duplicates(conds)
	print("----")
	for i in range(len(conds)):
		print(conds[i])
	return conds



		
@njit(nogil=True,fastmath=True, cache=True, locals={"ZERO":u1, "VISIT":u1, "VISITED": u1, "_n":i4})
def predict_tree(tree,xb,xc,pred_choice_enum,positive_class=0,decode_classes=True):
	'''Predicts the class associated with an unlabelled sample using a fitted 
		decision/ambiguity tree'''
	ZERO, VISIT, VISITED = 0, 1, 2
	L = max(len(xb),len(xc))
	out = np.empty((L,),dtype=np.int64)
	y_uvs = tree.u_ys#_get_y_order(tree)
	for i in range(L):
		# Use a mask instead of a list to avoid repeats that can blow up
		#  if multiple splits are possible. Keep track of visited in case
		#  of loops (Although there should not be any loops).
		new_node_mask = np.zeros((len(tree.nodes),),dtype=np.uint8)
		new_node_mask[0] = 1
		node_inds = np.nonzero(new_node_mask==VISIT)[0]
		leafs = List()

		while len(node_inds) > 0:
			#Mark all node_inds as visited so we don't mark them for a revisit
			for ind in node_inds:
				new_node_mask[ind] = VISITED

			# Go through every node that has been queued for a visit. In a traditional
			#  decision tree there should only ever be one next node.
			for ind in node_inds:
				node = tree.nodes[ind]
				if(node.ttype == TreeTypes_NODE):
					# Test every split in the node. Again in a traditional decision tree
					#  there should only be one split per node.
					for s in node.split_data:
						split_on, ithresh, left, right, nan  = s[0],s[1],s[2],s[3],s[4]

						# Determine if this sample should feed right, left, or nan (if ternary)
						if(split_on < xb.shape[1]):
							# Binary case
							j = split_on 
							if(np.isnan(xb[i,j])):
								_n = nan
							elif(xb[i,j]):
								_n = right
							else:
								_n = left
						else:
							# Continous case
							thresh = np.int32(ithresh).view(np.float32)
							j = split_on-xb.shape[1] 

							if(np.isnan(xc[i,j])):
								_n = nan
							elif(xc[i,j] >= thresh):
								_n = right
							else:
								_n = left
						if(new_node_mask[_n] != VISITED): new_node_mask[_n] = VISIT
							
				else:
					leafs.append(node.counts)

			node_inds = np.nonzero(new_node_mask==VISIT)[0]
		# print(leafs)
		# Since the leaf that the sample ends up in is ambiguous for an ambiguity
		#   tree we need a subroutine that chooses how to classify the sample from the
		#   various leaves that it could end up in. 
		out_i = pred_choice_func(pred_choice_enum, leafs, positive_class)
		if(decode_classes):out_i = y_uvs[out_i]
		out[i] = out_i
	return out


######### Repr/Visualtization #########

def str_tree(tree):
	'''A string representation of a tree usable for the purposes of debugging'''
	
	print(tree)
	# l = ["TREE w/ classes: %s"%_get_y_order(tree)]
	l = ["TREE w/ classes: %s"%tree.u_ys]
	# node_offset = 1
	# while node_offset < tree[0]:
	for node in tree.nodes:
		# node_width = tree[node_offset]
		ttype, index, splits, counts = node.ttype, node.index, node.split_data, node.counts#_unpack_node(tree,node_offset)
		if(ttype == TreeTypes_NODE):
			s  = "NODE(%s) : " % (index)
			for split in splits:
				if(split[1] == 1): #<-A threshold of 1 means it's binary
					s += "(%s)[L:%s R:%s" % (split[0],split[2],split[3])
				else:
					thresh = np.int32(split[1]).view(np.float32)
					s += "(%s,<%s)[L:%s R:%s" % (split[0],thresh,split[2],split[3])
				s += "] " if(split[4] == -1) else ("NaN:" + str(split[4]) + "] ")
			l.append(s)
		else:
			s  = "LEAF(%s) : %s" % (index,counts)
			l.append(s)
		# node_offset += node_width
	return "\n".join(l)


def print_tree(tree):
	print(str_tree(tree))


tree_classifier_presets = {
	'decision_tree' : {
		'criterion' : 'gini',
		'split_choice' : 'single_max',
		'pred_choice' : 'majority',
		'positive_class' : 1,
		'sep_nan' : True,
		'cache_nodes' : False
	},
	'ambiguity_tree' : {
		'criterion' : 'gini',
		'split_choice' : 'all_max',
		'pred_choice' : 'pure_majority',
		'positive_class' : 1,
		'sep_nan' : True,
		'cache_nodes' : True
	}

}
		
class TreeClassifier(object):
	def __init__(self,preset_type='decision_tree', 
					  **kwargs):
		'''
		TODO: Finish docs
		kwargs:
			preset_type: Specifies the values of the other kwargs

			criterion: The name of the criterion function used 'entropy', 'gini', etc.
			split_choice: The name of the split choice policy 'all_max', etc.
			pred_choice: The prediction choice policy 'pure_majority_general' etc.
			positive_class: The integer id for the positive class (used in prediction)
			sep_nan: If set to True then use a ternary tree that treats nan's seperately 
		'''
		kwargs = {**tree_classifier_presets[preset_type], **kwargs}

		criterion, split_choice, pred_choice, positive_class, sep_nan, cache_nodes = \
			itemgetter('criterion','split_choice', 'pred_choice', 'positive_class',
			 'sep_nan', 'cache_nodes')(kwargs)

		g = globals()
		criterion_enum = g.get(f"CRITERION_{criterion}",None)
		split_enum = g.get(f"SPLIT_CHOICE_{split_choice}",None)
		pred_choice_enum = g.get(f"PRED_CHOICE_{pred_choice}",None)

		if(criterion_enum is None): raise ValueError(f"Invalid criterion {criterion}")
		if(split_enum is None): raise ValueError(f"Invalid split_choice {split_choice}")
		if(pred_choice_enum is None): raise ValueError(f"Invalid pred_choice {pred_choice}")
		self.positive_class = positive_class

		@njit(cache=True)
		def _fit(xb,xc,y,missing_values=None):	
			if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
			out =fit_tree(xb,xc,y,
					missing_values=missing_values,
					criterion_enum=literally(criterion_enum),
					split_enum=literally(split_enum),
					sep_nan=literally(sep_nan),
					cache_nodes=literally(cache_nodes)
				 )
			return out
		self._fit = _fit

		@njit(cache=True)
		def _predict(tree, xb, xc, missing_values=None, positive_class=1):	
			if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
			out =predict_tree(tree,xb,xc,
					pred_choice_enum=literally(pred_choice_enum),
					positive_class=positive_class,
					decode_classes=True
				 )
			return out
		self._predict = _predict
		self.tree = None
		
	def fit(self,xb,xc,y,missing_values=None):
		if(xb is None): xb = np.empty((0,0), dtype=np.bool)
		if(xc is None): xc = np.empty((0,0), dtype=np.float64)
		self.tree = self._fit(xb, xc, y, missing_values)

	def predict(self,xb,xc,positive_class=None):
		if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
		if(positive_class is None): positive_class = self.positive_class
		if(xb is None): xb = np.empty((0,0), dtype=np.bool)
		if(xc is None): xc = np.empty((0,0), dtype=np.float64)
		return self._predict(self.tree, xb, xc, positive_class)

	def __str__(self):
		return str_tree(self.tree)

	def as_conditions(self,positive_class=None, only_pure_leaves=False):
		if(positive_class is None): positive_class = self.positive_class
		return tree_to_conditions(self.tree, positive_class, only_pure_leaves)


@jit(cache=True)
def test_fit(x,y,missing_values=None):	
	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
	out =fit_tree(x,np.empty((0,0), dtype=np.float64),y,
			missing_values=missing_values,
			criterion_enum=1,
			split_enum=1,
			sep_nan=True
		 )
	return out



@jit(cache=True)
def test_Afit(x,y,missing_values=None):	
	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
	out =fit_tree(x,np.empty((0,0), dtype=np.float64),y,
			missing_values=missing_values,
			criterion_enum=1,
			split_enum=2,
			cache_nodes=True,
		 )
	return out


if(__name__ == "__main__"):

	xc = np.triu(np.ones((6,6)))
	# xc = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	# [10,10,10], #3
	# [1, 0, 5], #1
	# [1, 0, 4], #1
	# [1, 0, 7], #1
	# [0, 0, 1], #2
	# [0, 0, 1], #2
	# [0, 0, 1], #2
	# ],np.float64);

	xb = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	# [1, 1, 1], #3
	[1, 0, 1], #1
	[1, 0, 1], #1
	[1, 0, 1], #1
	[0, 0, 1], #2
	[0, 0, 1], #2
	[0, 0, 1], #2
	],np.bool);

	y = np.asarray([0,0,0,1,1,1],np.int64);
	
	missing_values = np.empty((0,2),np.int64)
	get_counts_impurities(xb, xc, y, missing_values, 1.0, CRITERION_gini, 2, True)

	xc = np.asarray([
	# [10,10,10], #3
	[1, 1, 5], #1
	[1, 0, 4], #1
	[1, 1, 7], #1
	[0, 0, 1], #2
	[0, 1, 1], #2
	[0, 0, 1], #2
	],np.float64);

	get_counts_impurities(xb, xc, y, missing_values, 1.0, CRITERION_gini, 2, True)


	xc = np.asarray([
	# [10,10,10], #3
	[1, 0, 5], #1
	[1, 0, 4], #1
	[1, 0, 7], #1
	[np.nan, 0, 1], #2
	[np.nan, 0, 1], #2
	[np.nan, 0, 1], #2
	],np.float64);

	get_counts_impurities(xb, xc, y, missing_values, 1.0, CRITERION_gini, 2, True)

	#Check that it can find optimal splits when they exist
	N = 10
	xc = np.asarray([np.arange(N)],np.float64).T
	xb = np.zeros((0,0),np.bool)
	for i in range(N+1):
		y = np.concatenate([np.zeros(i,dtype=np.int64),np.ones(N-i,dtype=np.int64)])
		out = get_counts_impurities(xb, xc, y, missing_values, 1.0, CRITERION_gini, 2, True)
		countsPS, impurities, thresholds = out
		# assert tot_impurities[0] == 0.0
		if(i ==0 or i == N): 
			#When the input is pure the threshold should be inf
			assert thresholds[0] == np.inf
			assert all(np.sum(countsPS[0],axis=1) == np.array([10,0,0]))
		else:
			assert thresholds[0] > i-1 and thresholds[0] < i
			assert all(np.sum(countsPS[0],axis=1) == np.array([i,N-i,0]))
	


		

		# print()
		print(countsPS[0], impurities[0], thresholds[0])


	# np.empty(())
	
	
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

	labels = np.asarray([3,1,1,1,2,2,2],np.int64);
	clf = SKTree.DecisionTreeClassifier()
	# my_bdt = ILPTree()
	my_AT = TreeClassifier()


	# nb_fit = my_bdt.nb_ilp_tree.fit
	# cc = CC("my_module")
	# # compile_template(fit_tree,{'criterion_func': gini,'split_chooser': choose_single_max,
	# # 	'sep_nan':False, 'cache_nodes':False,},cc,'TR(b1[:,:],u4[:])',globals())
	# compile_template(fit_tree,{'criterion_func': gini,'split_chooser': choose_all_max,
	# 	'sep_nan':False, 'cache_nodes':True,},cc,'i4[:](b1[:,:],u4[:])',globals())
	# cc.compile()
	# from my_module import fit_tree_gini_choose_all_max_False_True

	##Compiled 
	# cc = CC("my_module")
	# compile_template(fit_tree,{'criterion_func': gini,"split_chooser":choose_single_max},cc,'i4[:](b1[:,:],u8[:])',globals())
	# cc.compile()
	# from my_module import fit_tree_gini	
	# def c_bdt():
	# 	fit_tree_gini(data,labels)
	###
	N = 100
	def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

	def bdt():
		test_fit(data,labels)

	def At():
		test_Afit(data,labels)
		# binary_decision_tree(data,labels)
		# my_bdt.fit(data,labels)
		# nb_fit(data,labels,gini)

	def skldt():
		clf.fit(data,labels)

	def control():
		return 0
	
	
	# f = get_criterion_func('gini')
	print(numba.typeof(gini))
	print(numba.typeof(unique_counts(labels)))

	# print(numba.typeof({}))
	
	# print("control:", time_ms(control))
	# print("t1:", time_ms(t1))
	# print("t2:", time_ms(t2))
	# print("t3:", time_ms(t3))
	# print("t4:", time_ms(t4))
	# print("t5:", time_ms(t5))
	# print("t6:", time_ms(t6))

	# print("d_tree:   ", time_ms(bdt))
	# print("a_tree:   ", time_ms(At))
	# print("numba_c  ", time_ms(c_bdt))
	# print("sklearn: ", time_ms(skldt))

	# bdt()
	sorted_inds = np.argsort(labels)
	y_sorted = labels[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y_sorted);


	treeA = test_Afit(data,labels)
	tree = test_fit(data,labels)

	
	print_tree(treeA)
	print("___")
	print_tree(tree)


	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,0,0,0,0], #1
	[0,0,1,0,0,0], #1
	[0,1,1,0,0,0], #1
	[1,1,1,0,0,1], #2
	[0,1,1,1,1,0], #2
	[1,1,1,0,1,0], #2
	],np.bool);

	labels = np.asarray([1,1,1,2,2,2],np.int64);
	data = data[:,[1,0,2,3,4,5]]

	xc = np.empty((0,2), dtype=np.int64)
	# tree = test_fit(data[:,[1,0,2,3,4,5]],labels)
	# treeA = test_Afit(data[:,[1,0,2,3,4,5]],labels)
	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,xc,PRED_CHOICE_pure_majority,positive_class=1))
	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,xc,PRED_CHOICE_pure_majority,positive_class=1))
	# my_AT.fit(data,labels)
	# print("MY_AT",my_AT.predict(data))
	# print("MY_AT",my_AT)

	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0], #1
	[1,0], #1
	[0,1], #1
	[1,1], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int64);

	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,xc,PRED_CHOICE_pure_majority,positive_class=1))

	# print("___")
	# print_tree(treeA)
	# print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE_pure_majority,positive_class=1))


	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0], #1
	[1,0], #1
	[0,1], #1
	[1,1], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int64);
	missing_values = np.asarray([[1,0]],np.int64)

	tree = test_fit(data,labels,missing_values)
	treeA = test_Afit(data,labels,missing_values)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,xc,PRED_CHOICE_pure_majority,positive_class=1))

	# print("___")
	# print_tree(treeA)
	# print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE_pure_majority,positive_class=1))

	

	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[1,1], #1
	[1,1], #1
	[1,0], #1
	[1,0], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int64);

	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,xc,PRED_CHOICE_pure_majority,positive_class=1))

	# print("___")
	# print_tree(treeA)
	# print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE_pure_majority,positive_class=1))


	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[1,1], #1
	[1,1], #1
	[1,0], #1
	[1,0], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int64);
	missing_values = np.asarray([[2,1]],np.int64)

	tree = test_fit(data,labels,missing_values)
	treeA = test_Afit(data,labels,missing_values)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,xc,PRED_CHOICE_pure_majority,positive_class=1))


	# clf = SKTree.DecisionTreeClassifier()
	# clf.fit(data,labels)
	# print(clf.predict(data[[-1]])	)


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

a = {"obj1-contenteditable": False,
 "obj2-contenteditable": False,
 "obj3-contenteditable": False,
 "obj1-value": 5,
 "obj2-value": "",
 "obj3-value": "",
}

a = {"obj2-contenteditable": False,
 "obj3-contenteditable": True,
 "obj4-contenteditable": False,
 "obj2-value": "",
 "obj3-value": "",
 "obj4-value": 7,
}


# class DictVectorizer(object):
# 	def __init__(self):
# 		self.map = {}

# 	def vectorize(self,flat_state):
# 		# new_map = self.map
# 		for k in flat_state.keys():
# 			self.map[k] = len(self.map)

# 		out = np.array(len(self.map),dtype=np.float64)
# 		for k, v in flat_state.items():
# 			out[self.map[k]] = v

# 		return out


# dv = DictVectorizer()

# dv.vectorize(a)






















