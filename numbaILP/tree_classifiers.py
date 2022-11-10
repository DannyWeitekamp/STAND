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




optims1 = {"nogil":True,"fastmath":True,"inline":'never'}
optims0 = {"inline":'never'}


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


@njit(f8[::1](u4[:,:]),cache=True, **optims1)
def return_zero(counts):
	return np.zeros(counts.shape[0], dtype=np.float64)

@njit(f8[::1](u4[:,:]), cache=True, **optims1)
def gini(counts):
	out = np.empty(counts.shape[0], dtype=np.float64)
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

@njit(f8[::1](u4[:,:], u4, u4), cache=True, **optims1)
def prop_neg(counts, pos_ind, n_classes):
	out = np.empty(counts.shape[0], dtype=np.float64)
	for j in range(counts.shape[0]):
		total = 0; #use epsilon? 1e-10
		pos = counts[j,pos_ind]
		
		for i in range(counts.shape[1]):
			total += counts[j,i];

		s = 1.0;

		if(total > 0):
			s = (total-pos)/float(total);
		out[j] = s
		# print(j, total, pos, ":", total-pos, s)

	return out

@njit(f8[::1](u4[:,:], u4, u4), cache=True, **optims1)
def prop_pos(counts, pos_ind, n_classes):
	out = np.empty(counts.shape[0], dtype=np.float64)
	for j in range(counts.shape[0]):
		total = 0; #use epsilon? 1e-10
		pos = counts[j,pos_ind]
		
		for i in range(counts.shape[1]):
			total += counts[j,i];

		s = 1.0;

		if(total > 0):
			s = (pos)/float(total);
		out[j] = s
		# print(j, total, pos, ":", total-pos, s)

	return out

@njit(f8[::1](u4[:,:]), cache=True, **optims1)
def weighted_gini(counts):
	return np.sum(counts, axis=1)*gini(counts)


CRITERION_none = 0
CRITERION_gini = 1
CRITERION_entropy = 2
CRITERION_prop_neg = 3
CRITERION_prop_pos = 4
CRITERION_weighted_gini = 5


@njit(cache=True, inline='never')
def criterion_func(func_enum, counts, pos_ind, n_classes):
	if(func_enum == 0):
		return return_zero(counts)		
	elif(func_enum == 1):
		return gini(counts)
	elif(func_enum == 2):
		return return_zero(counts)
	elif(func_enum == 3):
		return prop_neg(counts, pos_ind, n_classes)
	elif(func_enum == 4):
		return prop_pos(counts, pos_ind, n_classes)
	elif(func_enum == 5):
		return weighted_gini(counts)

	return gini(counts)


######### TOTAL IMPURITY CHOOSER #######

@njit( f8(f8[:],),cache=True, inline='never')
def total_sum(impurities):
	return np.sum(impurities)

@njit( f8(f8[:],),cache=True, inline='never')
def total_min(impurities):
	return np.min(impurities)

@njit( f8(f8[:],),cache=True, inline='never')
def total_average(impurities):
	return np.sum(impurities) * .5

@njit( f8(f8[:],),cache=True, inline='never')
def total_inv_FOIL(impurities):
	raise NotImplementedError()


TOTAL_none = 0
TOTAL_sum = 1
TOTAL_min = 2
TOTAL_average = 3

@njit(cache=True, inline='never')
def total_func(func_enum, impurities):
	if(func_enum == 1):
		return total_sum(impurities)		
	elif(func_enum == 2):
		return total_min(impurities)
	elif(func_enum == 3):
		return total_average(impurities)
	return total_sum(impurities)

@njit(cache=True)
def total_func_multiple(func_enum, impurities):
	out = np.empty((len(impurities),), dtype=np.float64)
	for i,imps in enumerate(impurities):
		out[i] = total_func(func_enum, imps)
	return out



######### Split Choosers ##########

# class SPLIT_CHOICE(IntEnum):
# 	single_max = 1
# 	all_max = 2

# @njit(i4[::1](f8[:]),nogil=True,fastmath=True,cache=True)
@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_single_max(impurity_decrease):
	'''A split chooser that expands greedily by max impurity 
		(i.e. this is the chooser for typical decision trees)'''
	return np.asarray([np.argmax(impurity_decrease)])

# @njit(i4[::1](f8[:]),nogil=True,fastmath=True,cache=True)
@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_all_max(impurity_decrease):
	'''A split chooser that expands every decision tree 
		(i.e. this chooser forces to build whole ambiguity tree)'''
	m = np.max(impurity_decrease)
	return np.where(impurity_decrease==m)[0]

SPLIT_CHOICE_single_max = 1
SPLIT_CHOICE_all_max = 2


@njit(cache=True,inline='never')
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



@njit(nogil=True,fastmath=True,cache=True,inline='never')
def get_pure_counts(leaf_counts):
	pure_counts = List()
	for count in leaf_counts:
		if(np.count_nonzero(count) == 1):
			pure_counts.append(count)
	return pure_counts

@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_majority(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority of leaves.''' 
	predictions = np.empty((len(leaf_counts),),dtype=np.int32)
	for i,count in enumerate(leaf_counts):
		predictions[i] = np.argmax(count)
	c,u, inds = unique_counts(predictions)
	_i = np.argmax(c)
	return u[_i]

@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_pure_majority(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority pure of leaves.'''
	pure_counts = get_pure_counts(leaf_counts)
	leaf_counts = pure_counts if len(pure_counts) > 0 else leaf_counts
	return choose_majority(leaf_counts,positive_class)

@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_majority_general(leaf_counts,positive_class):
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return 1
	return 0

@njit(nogil=True,fastmath=True,cache=True,inline='never')
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

@njit(nogil=True,fastmath=True,cache=True,inline='never')
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
def counts_per_binary_split(xb, y_inds, n_classes):
	''' 
		Determines the number of elements of each class that would be in the resulting
		left, right and nan nodes if a split was made at each possible binary feature.
		Also outputs the index at which missing values stop being applicable to binary
		features.
	'''
	# miss_i, miss_j = -1, -1
	# if (len(missing_values) > 0):
	# 	miss_i, miss_j = missing_values[0]
	# miss_index = 1 #if(miss_j < xb.shape[1]) else 0
	

	counts = np.zeros((xb.shape[1], 2, n_classes),dtype=np.uint32);
	miss_counts = np.zeros((xb.shape[1], n_classes),dtype=np.uint32);
	# nan_counts = np.zeros((xb.shape[1], n_classes),dtype=np.uint32);
	# has_nan = np.zeros((xb.shape[1], n_classes),dtype=np.uint32);
	# has_nan = False

	#Go through in Fortran order
	for j in range(xb.shape[1]):
		for i in range(xb.shape[0]):
			x_ij = xb[i,j]
			if(x_ij):
				if(x_ij == 1):
					counts[j,1,y_inds[i]] += 1;	
				else:
					miss_counts[j,y_inds[i]] += 1;

			else:
				counts[j,0,y_inds[i]] += 1;	
			
				
				


	

	return counts, miss_counts






@njit(nogil=True,fastmath=True,cache=True)
def r_l_split(x, miss_mask):
	'''Similar to argwhere applied 3 times each for 0,1 and nan, but does all
		three at once.'''
	nl,nr = 0,0
	l = np.empty(x.shape,np.uint32)
	r = np.empty(x.shape,np.uint32)
	# n = np.empty(x.shape,np.uint32)

	# next_missing = missing[0] if len(missing) > 0 else -1
	# m_ind = 1
	
	for i in range(len(x)):
		x_i = x[i]

		# if(i == next_missing):
		if(miss_mask[i]):
			# n[nn] = i
			# nn += 1
			# r[nr] = i
			# nr += 1

			#Missing values always go left
			l[nl] = i
			nl += 1
			# if(m_ind < len(missing)):
			# 	next_missing = missing[m_ind]
				# m_ind += 1
		# elif(sep_nan and x_i == 255):
		# 	n[nn] = i
		# 	nn += 1
		else:
			if(x[i]):
				r[nr] = i
				nr += 1
			else:
				l[nl] = i
				nl += 1
	return l[:nl], r[:nr]


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
	('op_enum', u1),
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
@njit(cache=True, locals={"NODE":i4,"LEAF":i4,'node':i4},inline='never')
def new_node(locs, split, op, new_inds, impurities, countsPS,ind):
	node_dict,nodes,new_contexts,cache_nodes = locs
	NODE, LEAF = i4(1), i4(2) #np.array(1,dtype=np.int32).item(), np.array(2,dtype=np.int32).item()
	node = i4(-1)
	if (cache_nodes): node= akd_get(node_dict,new_inds)
	if(node == -1):
		node = i4(len(nodes))
		if(cache_nodes): akd_insert(node_dict,new_inds,node)
		ms_impurity = impurities[split,ind].item()
		if(ms_impurity > 0.0):
			nodes.append(TreeNode(NODE,node,op, List.empty_list(i4_arr),countsPS[split,ind]))
			new_contexts.append(SplitContext(new_inds,
				ms_impurity,countsPS[split,ind], node))
		else:
			nodes.append(TreeNode(LEAF,node,op, List.empty_list(i4_arr),countsPS[split,ind]))
	return node


OP_NOP = u1(0)
OP_GE = u1(1)
OP_LT = u1(2) 
OP_ISNAN = u1(3)


@njit(cache=True)
def get_counts_impurities(xb, xc, y, miss_mask, base_impurity, counts, criterion_enum, total_enum, pos_ind, n_classes, sep_nan):
	#NOTE: This function assumes that the elements [i,j] of missing_values is sorted by 'j'  
	n_b, n_c = xb.shape[1], xc.shape[1]
	countsPS = np.empty((n_b+n_c, 2, n_classes),dtype=np.uint32)
	impurities = np.empty((n_b+n_c, 2),dtype=np.float64)
	ops = np.empty((n_b+n_c,),dtype=np.uint8)
	# Handle binary case
	countsPS_n_b, miss_countsPS = counts_per_binary_split(xb, y, n_classes)
	countsPS[:n_b] = countsPS_n_b
	flat_impurities = criterion_func(criterion_enum, countsPS_n_b.reshape((-1,n_classes)), pos_ind, n_classes)
	impurities[:n_b] = flat_impurities.reshape((n_b,2))

	ops[:n_b] = OP_GE

	# Throw missing values into the left bin
	for j in range(n_b):
		countsPS[j,0] = countsPS[j,0] + miss_countsPS[j]
	
	
	# Handle continous case	
	thresholds = np.empty((n_c,),dtype=np.float64)
	
	
	is_pure = np.sum(counts > 0) <= 1
	if(is_pure):
		# If this node is pure then just return placeholder info
		for j in range(n_c):
			countsPS[j] = np.zeros((2, n_classes),dtype=np.uint32)
			countsPS[j,0] = counts
			thresholds[n_b +j] = np.inf
			impurities[n_b +j] = base_impurity
			ops[n_b+j] = OP_GE
	else:
		# Otherwise we need to find the best threshold to split on per feature
		for j in range(n_c):
			# Generate and indicies along i that excludes missing values
			miss_counts = np.zeros((n_classes,),dtype=np.uint32)

			if(miss_mask.shape[1] > 0):
				# If the miss_mask exists count the fill in miss_counts and slice out 
				#  any missing values
				mm_j = miss_mask[:,j]
				for i,tr in enumerate(mm_j):
					if(tr): miss_counts[y[i]] += 1
				non_miss_j = ~mm_j

				# Select all non missing features and labels for candidate split j 
				xc_j = xc[non_miss_j,j]
				y_j = y[non_miss_j]
			else:
				xc_j = xc[:,j]
				y_j = y
			
			
			# Sort by feature
			srt_inds = np.argsort(xc_j)
			xc_j = xc_j[srt_inds]
			y_j = y_j[srt_inds]			

			# print("xc_j",xc_j)
			# print("y_j",y_j)

			# NaN's end up at the end of a sort (in numpy), find where they start 
			nan_start = len(xc_j)
			for i in range(len(xc_j)-1,-1,-1):
				if(not np.isnan(xc_j[i])): break
				nan_start = i

			#Find counts for NaN only bin
			has_nan = nan_start != len(xc_j)
			nan_counts = np.zeros((n_classes,),dtype=np.uint32)
			for i in range(nan_start,len(xc_j)):
				nan_counts[y_j[i]] += 1

			# print("nan_start",nan_start)
			
			# Find left(0) and right(1) cumulative counts for splitting at each possible threshold
			#  i.e figure out what child counts looks like if we put the threshold inbetween
			#  each of the sorted feature pairs.			
			cum_counts = np.zeros((nan_start+1, 2, n_classes),dtype=np.uint32)
			for i in range(nan_start):
				y_ij = y_j[i]
				cum_counts[i+1, 0] = cum_counts[i, 0]
				cum_counts[i+1, 0, y_ij] += 1

			cum_counts[:,1] = (counts-(miss_counts+nan_counts)) - cum_counts[:,0]

			# print("cum_counts", cum_counts)
			
			#Find all 'i' s.t. xc_j[i] != xc_j[i-1] (i.e. candidate pairs for threshold)
			thresh_inds, c = np.empty((nan_start,),dtype=np.uint32), 0
			for i in range(1, nan_start):
				if(xc_j[i] != xc_j[i-1]):
					thresh_inds[c] = i
					c += 1
					
			# If every value is the same then just use i=0
			best_impurity = np.zeros((2,),dtype=np.float64)
			best_impurity[0] = base_impurity
			best_impurity[1] = 1.0#np.inf
			best_total_impurity, best_counts, best_op = np.inf, cum_counts[-1], OP_GE

			
			if(c > 0):
				# If the features are not all the same find the best threshold
				thresh_inds = thresh_inds[:c]
				split_counts = best_ind = -1
				for t_i in thresh_inds:
					# We need to find the best of three choices for handling NaNs 
					#  NOTE: can't just use >= since for all t (NaN >= t) == 0 
				 	#  1 : (Nan|N vs Y) : x >= thresh
					#  2 : (Nan|Y vs N) : x < thresh 
					#  3 : (Y|N vs Nan) : np.isnan(x)

					# Check (1 and 2) on every possible threshold
					#  to see what leads to the smallest impurity
					if(sep_nan and has_nan):
						# Build counts for '<' and '>=", always putting NaNs in left
						c_lt = np.empty((2, n_classes),dtype=np.uint32)
						c_lt[0] = cum_counts[t_i,1] + nan_counts
						c_lt[1] = cum_counts[t_i,0]
						c_ge = np.empty((2, n_classes),dtype=np.uint32)
						c_ge[0] = cum_counts[t_i,0] + nan_counts
						c_ge[1] = cum_counts[t_i,1]

						# Figure out which operation is better in terms of total impurity
						impurity_lt = criterion_func(criterion_enum, c_lt, pos_ind, n_classes)
						impurity_ge = criterion_func(criterion_enum, c_ge, pos_ind, n_classes)
						total_impurity_lt = total_func(total_enum, impurity_lt) 
						total_impurity_ge = total_func(total_enum, impurity_ge) 
						if(total_impurity_lt < total_impurity_ge):
							impurity = impurity_lt
							total_impurity = total_impurity_lt
							op = OP_LT 
							split_counts = c_lt
						else:
							impurity = impurity_ge
							total_impurity = total_impurity_ge
							op = OP_GE 
							split_counts = c_ge
					else:
						# When there are no NaNs or if we ignore them fallback on ">="
						split_counts = cum_counts[t_i]
						impurity = criterion_func(criterion_enum, split_counts, pos_ind, n_classes)
						total_impurity = total_func(total_enum, impurity)
						op = OP_GE
					
					if(total_impurity < best_total_impurity):
						# If this is the best theshold point so far mark it as such
						best_impurity, best_total_impurity = impurity, total_impurity
						best_ind, best_op = t_i, op
						best_counts = split_counts

				thresh = (xc_j[best_ind-1] + xc_j[best_ind]) / 2.0 #if best_ind != 0 else np.inf
			else:
				# Otherwise just use a placeholder threshold
				best_op = OP_GE
				thresh = np.inf
			
			#See if using np.is_nan() would produce better results
			if(sep_nan and has_nan):
				#Left is non_NaN right is NaN
				is_nan_counts = np.empty((2, n_classes),dtype=np.uint32)
				is_nan_counts[0] = counts-(nan_counts+miss_counts)
				is_nan_counts[1] = nan_counts

				is_nan_impurity = criterion_func(criterion_enum, is_nan_counts,pos_ind, n_classes)
				is_nan_total_impurity = total_func(total_enum, is_nan_impurity)
				if(is_nan_total_impurity < best_total_impurity):
					best_impurity, best_total_impurity = is_nan_impurity, is_nan_total_impurity
					best_op = OP_ISNAN
					best_counts = is_nan_counts

			#Fill in outputs for candidate split j
			impurities[n_b+j,:2] = best_impurity
			thresholds[j] = thresh
			countsPS[n_b+j,:2] = best_counts#
			ops[n_b+j] = best_op
			
			# Even though missing values are ignored in impurity calculations 
			#   the total counts still need to be correct. Throw them in left bin.
			countsPS[n_b+j, 0] = countsPS[n_b+j, 0] + miss_counts 

			
	# print(countsPS)
	# print("impurities")
	# print(impurities)
	# print("thresholds",thresholds)
	# print("ops",ops)
	return countsPS, impurities, thresholds, ops




# The problem is where do I put the NaN values:
#  -They are different from missing
#  -There are three possibilities (Nan vs Y|N), (Nan|Y vs N), (Nan|N vs Y)
#  -Should handle first after (sep_nan) and other two in (c > 0)
#  -So it's


@njit(cache=True)
def inf_gain(x_bin, x_cont, y, miss_mask, ft_weights, criterion_enum, total_enum, positive_class=1):
	sorted_inds = np.argsort(y)
	x_bin_sorted = x_bin[sorted_inds]
	x_cont_sorted = x_cont[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);

	#Find the y_ind value associated with the positive class
	pos_ind = 0
	for i, u_ys_i in enumerate(u_ys):
		if(u_ys_i == positive_class): pos_ind = i

	n_classes = len(u_ys)
	impurity = criterion_func(criterion_enum, np.expand_dims(counts,0), pos_ind, n_classes)[0]

	c_xb, c_xc, c_y = x_bin_sorted, x_cont_sorted, y_inds
	c_mm = miss_mask 

	countsPS, impurities, thresholds, ops =  \
	get_counts_impurities(c_xb, c_xc, c_y, c_mm, impurity, counts,
							criterion_enum, total_enum, pos_ind, n_classes, True)
	# print("BI:", c.impurity)
	# print("IMP:", impurities)
	# print(countsPS, impurities, thresholds, ops)
	#Sum of new impurities of left and right side of split
	total_split_impurity = total_func_multiple(total_enum, impurities)
	# if(sep_nan): total_split_impurity += impurities[:,2]
	impurity_decrease = impurity - (total_split_impurity);
	return np.where(impurity_decrease > 0, impurity_decrease, 0.0)




@njit(cache=True, locals={"ZERO":i4,"NODE":i4,"LEAF":i4,"n_nodes":i4,"node_l":i4,"node_r":i4,"node_n":i4,"split":i4})
def fit_tree(x_bin, x_cont, y, miss_mask, ft_weights, criterion_enum, total_enum, split_enum, criterion_enum2=0, total_enum2=0, positive_class=1, sep_nan=False, cache_nodes=False):
	'''Fits a decision/ambiguity tree'''

	#ENUMS definitions necessary if want to use 32bit integers since literals default to 64bit
	ZERO, NODE, LEAF = 0, 1, 2
	sorted_inds = np.argsort(y)
	x_bin_sorted = x_bin[sorted_inds]
	x_cont_sorted = x_cont[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);

	#Find the y_ind value associated with the positive class
	pos_ind = 0
	for i, u_ys_i in enumerate(u_ys):
		if(u_ys_i == positive_class): pos_ind = i

	n_classes = len(u_ys)
	impurity = criterion_func(criterion_enum, np.expand_dims(counts,0), pos_ind, n_classes)[0]

	contexts = List.empty_list(SC)
	contexts.append(SplitContext(np.arange(0,len(y),dtype=np.uint32),impurity,counts,ZERO))

	node_dict = Dict.empty(u4,BE_List)
	nodes = List.empty_list(TN)
	nodes.append(TreeNode(NODE,ZERO,OP_NOP,List.empty_list(i4_arr),counts))
	while len(contexts) > 0:
		new_contexts = List.empty_list(SC)
		locs = (node_dict,nodes,new_contexts,cache_nodes)
		for i in range(len(contexts)):
			c = contexts[i]
			# print("c_ind", c.inds)
			c_xb, c_xc, c_y = x_bin_sorted[c.inds], x_cont_sorted[c.inds], y_inds[c.inds]
			c_mm = miss_mask if(miss_mask.shape[1] == 0) else miss_mask[c.inds]

			# c_xb, c_y = x_sorted[c.inds], y_inds[c.inds]
			#Bad solution, 
			# ms_v, k = np.empty((len(c.inds),2), dtype=np.int64), 0
			# for miss_i, miss_j in missing_values:
			# 	if(miss_i in c.inds):
			# 		ms_v[k,0] = miss_i
			# 		ms_v[k,1] = miss_j
			# 		k += 1
			# ms_v = ms_v[:k]
			# missing = np.argwhere(missing_values[:,0] == split)[:,0]
			# countsPS = counts_per_split(c_x, c_y, n_classes, missing_values, sep_nan)
			# # print("M PS:", missing_values, "\n", countsPS)
			# flat_impurities = criterion_func(criterion_enum,countsPS.reshape((-1,countsPS.shape[2])))
			# impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))
			# def get_counts_impurities(xb, xc, y, miss_mask, base_impurity, counts, criterion_enum, total_enum, pos_ind, n_classes, sep_nan):
			countsPS, impurities, thresholds, ops =  \
				get_counts_impurities(c_xb, c_xc, c_y, c_mm, c.impurity, c.counts,
										criterion_enum, total_enum, pos_ind, n_classes, sep_nan)
			# print("BI:", c.impurity)
			# print("IMP:", impurities)
			# print(countsPS, impurities, thresholds, ops)
			#Sum of new impurities of left and right side of split
			total_split_impurity = total_func_multiple(total_enum, impurities)
			# if(sep_nan): total_split_impurity += impurities[:,2]
			impurity_decrease = c.impurity - (total_split_impurity);
			# print("impurity_decrease", impurity_decrease)
			# print("ft_weights", ft_weights)
			if(len(ft_weights) > 0): impurity_decrease = impurity_decrease * ft_weights
			# print("impurity_decrease", impurity_decrease)
			splits = split_chooser(split_enum, impurity_decrease)

			# print("ENUMS", criterion_enum2, total_enum2)
			if(criterion_enum2 and total_enum2 and
				np.max(impurity_decrease) <= 0.0 and c.impurity > 0.0):

				print("OLD SPLIT", splits, np.max(impurity_decrease))
				sec_c_impurity = criterion_func(criterion_enum2, c.counts.reshape(1,n_classes), pos_ind, n_classes)
				flat_impurities = criterion_func(criterion_enum2, countsPS.reshape((-1,n_classes)), pos_ind, n_classes)
				sec_impurities = flat_impurities.reshape((-1,2))

				total_split_impurity = total_func_multiple(total_enum2, sec_impurities)
				impurity_decrease = sec_c_impurity - (total_split_impurity);
				splits = split_chooser(split_enum, impurity_decrease)
				print("NEW SPLIT", splits, np.max(impurity_decrease))
			# print("splits",splits)
			# print("---------")
			# print("ops", ops)
			# print("---------")
			for j in range(len(splits)):
				split = splits[j]

				if(impurity_decrease[split] <= 0.0):
					nodes[c.parent_node]=TreeNode(LEAF,c.parent_node,OP_NOP,List.empty_list(i4_arr),c.counts)
				else:
					if (split < c_xb.shape[1]):
						op = OP_GE
						split_miss_mask = c_xb[:,split] > 1
						
						# print("c_xb",c_xb)
						mask = c_xb[:,split] == 1
						# print("bmask", mask)

					else: 
						
						op = ops[split]
						split_miss_mask = c_mm[:,split]

						# print("c_xc",c_xc)
						split_slice = c_xc[:,split-c_xb.shape[1]]
						# print("split_slice",c_xc)
						if(op == OP_GE):
							mask = (split_slice >= thresholds[split-c_xb.shape[1]])#.astype(np.uint8)
						elif(op == OP_LT):
							mask = (split_slice < thresholds[split-c_xb.shape[1]])#.astype(np.uint8)
						elif(op == OP_ISNAN):
							mask = np.isnan(split_slice)#.astype(np.uint8)
						# print("cmask", mask)
						# print(op)

						# n_mask = np.isnan(c_xc[:,split-c_xb.shape[1]]).astype(np.uint8)
						
						# mask = np.where(n_mask, u1(255), mask)
						# mask[n_mask] = 255
					# print("mask",mask)
					# print("split_miss_mask",split_miss_mask)
					# missing = np.argwhere(missing_values[:,1] == split)[:,0]
					# print("missing", split, missing)
					node_l, node_r = -1, -1
					# print("mask", mask)
					new_inds_l, new_inds_r = r_l_split(mask, split_miss_mask)
					# print("new_inds",c.inds, new_inds_l, new_inds_r)
					new_inds_l, new_inds_r = c.inds[new_inds_l], c.inds[new_inds_r]
					# print("new_inds",c.inds, new_inds_l, new_inds_r)
					# locs = (node_dict, nodes,new_contexts, cache_nodes)
					#New node for left.
					node_l = new_node(locs, split, OP_NOP, new_inds_l, impurities,countsPS, literally(0))

					#New node for right.
					node_r = new_node(locs, split, OP_NOP, new_inds_r, impurities,countsPS, literally(1))

					# #New node for NaN values.
					# if(sep_nan and len(new_inds_n) > 0):
					# 	node_n = new_node(locs,split,new_inds_n, impurities,countsPS,literally(2))
					
					#If is continous bitcast threshold to an i4 else set to 1 i.e. 1e-45
					thresh = np.float32(thresholds[split-c_xb.shape[1]]).view(np.int32) if split >= c_xb.shape[1] else 1
					split_data = np.array([split, thresh, node_l, node_r, -1],dtype=np.int32)
					nodes[c.parent_node].split_data.append(split_data)
					nodes[c.parent_node].op_enum = op

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
@njit(cache=True,inline='never')
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

@njit(cache=True,inline='never')
def _indexOf(tree,node_offset):
	'''Takes a tree encoded with encode_tree and the offset where a nodes is and returns
	   just the index of the node.'''
	return tree[node_offset+2]


@njit(cache=True,inline='never')
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

@njit(cache=True)
def exec_op(op, val, thresh):
	if(op == OP_GE):
		return val >= thresh
	elif(op == OP_LT):
		return val < thresh
	elif(op == OP_ISNAN):
		return np.isnan(val)
	return True




		
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
				op = node.op_enum
				if(node.ttype == TreeTypes_NODE):
					# Test every split in the node. Again in a traditional decision tree
					#  there should only be one split per node.
					for s in node.split_data:
						split_on, ithresh, left, right, nan  = s[0],s[1],s[2],s[3],s[4]

						# Determine if this sample should feed right, left, or nan (if ternary)
						if(split_on < xb.shape[1]):
							# Binary case
							j = split_on 
							if(xb[i,j]):
								_n = right
							else:
								_n = left
						else:
							# Continous case
							thresh = np.int32(ithresh).view(np.float32)
							j = split_on-xb.shape[1] 

							if(exec_op(op,xc[i,j],thresh)):
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

def str_op(op_enum):
	if(op_enum == OP_LT):
		return "<"
	elif(op_enum == OP_GE):
		return ">="
	elif(op_enum == OP_ISNAN):
		return "isNaN"
	else:
		return ""

def str_tree(tree):
	'''A string representation of a tree usable for the purposes of debugging'''
	
	# l = ["TREE w/ classes: %s"%_get_y_order(tree)]
	l = ["TREE w/ classes: %s"%tree.u_ys]
	# node_offset = 1
	# while node_offset < tree[0]:
	for node in tree.nodes:
		# node_width = tree[node_offset]
		ttype, index, splits, counts = node.ttype, node.index, node.split_data, node.counts#_unpack_node(tree,node_offset)
		op = node.op_enum
		if(ttype == TreeTypes_NODE):
			s  = "NODE(%s) : " % (index)
			for split in splits:
				if(split[1] == 1): #<-A threshold of 1 means it's binary
					s += "(%s)[L:%s R:%s" % (split[0],split[2],split[3])
				else:
					thresh = np.int32(split[1]).view(np.float32)
					instr = str_op(op)+str(thresh) if op != OP_ISNAN else str_op(op)
					s += "(%s,%s)[L:%s R:%s" % (split[0],instr,split[2],split[3])
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
		'total_func' : 'sum',
		'split_choice' : 'single_max',
		'pred_choice' : 'majority',
		'positive_class' : 1,
		"secondary_criterion" : 0,
		"secondary_total_func" : 0,
		'sep_nan' : True,
		'cache_nodes' : False
	},
	'decision_tree_weighted_gini' : {
		'criterion' : 'weighted_gini',
		'total_func' : 'sum',#sum', 
		'split_choice' : 'single_max',
		'pred_choice' : 'majority',
		'positive_class' : 1,
		"secondary_criterion" : 0,
		"secondary_total_func" : 0,
		'sep_nan' : True,
		'cache_nodes' : False
	},
	'decision_tree_w_greedy_backup' : {
		'criterion' : 'gini',
		'total_func' : 'sum',#sum', 
		'split_choice' : 'single_max',
		'pred_choice' : 'majority',
		'positive_class' : 1,
		"secondary_criterion" : 'prop_neg',
		"secondary_total_func" : 'min',
		'sep_nan' : True,
		'cache_nodes' : False
	},
	'ambiguity_tree' : {
		'criterion' : 'weighted_gini',
		'total_func' : 'sum',
		'split_choice' : 'all_max',
		'pred_choice' : 'pure_majority',
		'positive_class' : 1,
		"secondary_criterion" : 0,
		"secondary_total_func" : 0,
		'sep_nan' : True,
		'cache_nodes' : True
	},
	'greedy_cover_tree' : {
		'criterion' : 'prop_neg',
		'total_func' : 'min',
		'split_choice' : 'single_max',
		'pred_choice' : 'majority',
		"secondary_criterion" : 0,
		"secondary_total_func" : 0,
		'positive_class' : 1,
		'sep_nan' : True,
		'cache_nodes' : False
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
			total_func: The function for combining the impurities for two splits, default 'sum'.
			split_choice: The name of the split choice policy 'all_max', etc.
			pred_choice: The prediction choice policy 'pure_majority_general' etc.
			secondary_criterion: The name of the secondary criterion function used only if 
			  no split can be found with the primary impurity.
			secondary_total_func: The name of the secondary total_func, defaults to 'sum'
			positive_class: The integer id for the positive class (used in prediction)
			sep_nan: If set to True then use a ternary tree that treats nan's seperately 
		'''
		kwargs = {**tree_classifier_presets[preset_type], **kwargs}

		criterion, total_func, split_choice, pred_choice, secondary_criterion, \
		 secondary_total_func, positive_class, sep_nan, cache_nodes = \
			itemgetter('criterion', 'total_func', 'split_choice', 'pred_choice', 
				"secondary_criterion", 'secondary_total_func', 'positive_class',
				'sep_nan', 'cache_nodes')(kwargs)

		g = globals()
		criterion_enum = g.get(f"CRITERION_{criterion}",None)
		total_enum = g.get(f"TOTAL_{total_func}",None)
		split_enum = g.get(f"SPLIT_CHOICE_{split_choice}",None)
		pred_choice_enum = g.get(f"PRED_CHOICE_{pred_choice}",None)
		criterion_enum2 = g.get(f"CRITERION_{secondary_criterion}",0)
		total_enum2 = g.get(f"TOTAL_{secondary_total_func}", TOTAL_sum)

		if(criterion_enum is None): raise ValueError(f"Invalid criterion {criterion}")
		if(total_enum is None): raise ValueError(f"Invalid criterion {total_func}")
		if(split_enum is None): raise ValueError(f"Invalid split_choice {split_choice}")
		if(pred_choice_enum is None): raise ValueError(f"Invalid pred_choice {pred_choice}")
		self.positive_class = positive_class

		@njit(cache=True)
		def _fit(xb,xc,y,miss_mask,ft_weights):	
			out =fit_tree(xb,xc,y,miss_mask,
					ft_weights=ft_weights,
					# missing_values=missing_values,
					criterion_enum=literally(criterion_enum),
					total_enum=literally(total_enum),
					split_enum=literally(split_enum),
					positive_class=positive_class,
					criterion_enum2=literally(criterion_enum2),
					total_enum2=literally(total_enum2),
					sep_nan=literally(sep_nan),
					cache_nodes=literally(cache_nodes)
				 )
			return out
		self._fit = _fit

		@njit(cache=True)
		def _inf_gain(xb,xc,y,miss_mask,ft_weights):	
			out =inf_gain(xb,xc,y,miss_mask,
					ft_weights=ft_weights,
					# missing_values=missing_values,
					criterion_enum=literally(criterion_enum),
					total_enum=literally(total_enum),
					positive_class=positive_class,
				 )
			return out
		self._inf_gain = _inf_gain

		@njit(cache=True)
		def _predict(tree, xb, xc, positive_class):	
			out =predict_tree(tree,xb,xc,
					pred_choice_enum=literally(pred_choice_enum),
					positive_class=positive_class,
					decode_classes=True
				 )
			return out
		self._predict = _predict
		self.tree = None
		
	def fit(self,xb,xc,y,miss_mask=None, ft_weights=None):
		if(xb is None): xb = np.empty((0,0), dtype=np.uint8)
		if(xc is None): xc = np.empty((0,0), dtype=np.float64)
		if(miss_mask is None): miss_mask = np.zeros_like(xc, dtype=np.bool)
		if(ft_weights is None): ft_weights = np.empty(xb.shape[1]+xc.shape[1], dtype=np.float64)
		xb = xb.astype(np.uint8)
		xc = xc.astype(np.float64)
		y = y.astype(np.int64)
		miss_mask = miss_mask.astype(np.bool)
		ft_weights = ft_weights.astype(np.float64)
		# assert miss_mask.shape == xc.shape
		self.tree = self._fit(xb, xc, y, miss_mask, ft_weights)

	def inf_gain(self,xb,xc,y,miss_mask=None, ft_weights=None):
		if(xb is None): xb = np.empty((0,0), dtype=np.uint8)
		if(xc is None): xc = np.empty((0,0), dtype=np.float64)
		if(miss_mask is None): miss_mask = np.zeros_like(xc, dtype=np.bool)
		if(ft_weights is None): ft_weights = np.empty(xb.shape[1]+xc.shape[1], dtype=np.float64)
		xb = xb.astype(np.uint8)
		xc = xc.astype(np.float64)
		y = y.astype(np.int64)
		miss_mask = miss_mask.astype(np.bool)
		ft_weights = ft_weights.astype(np.float64)
		# assert miss_mask.shape == xc.shape
		return self._inf_gain(xb, xc, y, miss_mask, ft_weights)


	def predict(self,xb,xc,positive_class=None):
		if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
		if(positive_class is None): positive_class = self.positive_class
		if(xb is None): xb = np.empty((0,0), dtype=np.uint8)
		if(xc is None): xc = np.empty((0,0), dtype=np.float64)
		xb = xb.astype(np.uint8)
		xc = xc.astype(np.float64)
		return self._predict(self.tree, xb, xc, positive_class)

	def __str__(self):
		return str_tree(self.tree)

	def as_conditions(self,positive_class=None, only_pure_leaves=False):
		if(positive_class is None): positive_class = self.positive_class
		return tree_to_conditions(self.tree, positive_class, only_pure_leaves)


@jit(cache=True)
def _test_fit(x,y,missing_values=None):	
	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
	out =fit_tree(x,np.empty((0,0), dtype=np.float64),y,
			missing_values=missing_values,
			criterion_enum=1,
			split_enum=1,
			sep_nan=True
		 )
	return out



@jit(cache=True)
def _test_Afit(x,y,missing_values=None):	
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
		_test_fit(data,labels)

	def At():
		_test_Afit(data,labels)
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


	treeA = _test_Afit(data,labels)
	tree = _test_fit(data,labels)

	
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
	# tree = _test_fit(data[:,[1,0,2,3,4,5]],labels)
	# treeA = _test_Afit(data[:,[1,0,2,3,4,5]],labels)
	tree = _test_fit(data,labels)
	treeA = _test_Afit(data,labels)
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

	tree = _test_fit(data,labels)
	treeA = _test_Afit(data,labels)
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

	tree = _test_fit(data,labels,missing_values)
	treeA = _test_Afit(data,labels,missing_values)
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

	tree = _test_fit(data,labels)
	treeA = _test_Afit(data,labels)
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

	tree = _test_fit(data,labels,missing_values)
	treeA = _test_Afit(data,labels,missing_values)
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






















