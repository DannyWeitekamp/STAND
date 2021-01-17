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
def counts_per_split(start_counts, x, y_inds, missing_values, sep_nan=False):
	''' 
		Determines the number of elements of each class that would be in the resulting
		left, right and nan nodes if a split was made at each possible feature.
	'''
	miss_i, miss_j = -1, -1
	if (len(missing_values) > 0):
		miss_i, miss_j = missing_values[0]
	miss_index = 1

	counts = np.zeros((x.shape[1],2+sep_nan,len(start_counts)),dtype=np.uint32);
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(i == miss_i and j == miss_j):
				# counts[j,0,y_inds[i]] += 1;	
				# counts[j,1,y_inds[i]] += 1;	
				miss_i, miss_j = missing_values[miss_index]
				miss_index += 1
			elif(sep_nan and np.isnan(x[i,j])):
				counts[j,2,y_inds[i]] += 1;	
			else:
				if(x[i,j]):
					counts[j,1,y_inds[i]] += 1;	
				else:
					counts[j,0,y_inds[i]] += 1;	
	return counts;






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
	left -- For each split in split_on the index of the node to the left
	right -- For each split in split_on the index of the node to the right
	nan -- For each split in split_on the index of the node in the nan slot
	counts -- If is a leaf node the number of samples of each class falling in it
'''
TreeNode = namedtuple("TreeNode",['ttype','index','split_data','counts'])
TN = NamedTuple([i4,i4,ListType(i4[:]),u4[::1]],TreeNode)


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


######### Fit #########

#NOTE: new_node is probably commented out in fit_tree and replaced by an inline implementation
#	numba's inlining isn't quite mature enough to not take a slight performance hit.
@njit(cache=True,locals={"NODE":i4,"LEAF":i4,'node':i4},inline='always')
def new_node(locs,split,new_inds, impurities,countsPS,ind):
	node_dict,nodes,new_contexts,cache_nodes = locs
	NODE, LEAF = np.array(1,dtype=np.int32).item(), np.array(2,dtype=np.int32).item()
	node = -1
	if (cache_nodes): node= akd_get(node_dict,new_inds)
	if(node == -1):
		node = np.array(len(nodes),dtype=np.int32).item()
		if(cache_nodes): akd_insert(node_dict,new_inds,node)
		ms_impurity = impurities[split,ind].item()
		if(ms_impurity > 0.0):
			nodes.append(TreeNode(NODE,node,List.empty_list(i4_arr),countsPS[split,ind]))
			new_contexts.append(SplitContext(new_inds,
				ms_impurity,countsPS[split,ind], node))
		else:
			nodes.append(TreeNode(LEAF,node,List.empty_list(i4_arr),countsPS[split,ind]))
	return node

@njit(cache=True, locals={"ZERO":i4,"NODE":i4,"LEAF":i4,"n_nodes":i4,"node_l":i4,"node_r":i4,"node_n":i4,"split":i4})
def fit_tree(x, y, missing_values, criterion_enum, split_enum, sep_nan=False, cache_nodes=False):
	'''Fits a decision/ambiguity tree'''

	#ENUMS definitions necessary if want to use 32bit integers since literals default to 64bit
	ZERO, NODE, LEAF = 0, 1, 2

	sorted_inds = np.argsort(y)
	x_sorted = x[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);
	impurity = criterion_func(criterion_enum,np.expand_dims(counts,0))[0]

	contexts = List.empty_list(SC)
	contexts.append(SplitContext(np.arange(0,len(x),dtype=np.uint32),impurity,counts,ZERO))

	node_dict = Dict.empty(u4,BE_List)
	nodes = List.empty_list(TN)
	nodes.append(TreeNode(NODE,ZERO,List.empty_list(i4_arr),counts))
	

	while len(contexts) > 0:
		new_contexts = List.empty_list(SC)
		# locs = (node_dict,nodes,new_contexts,cache_nodes)
		for i in range(len(contexts)):
			c = contexts[i]
			c_x, c_y = x_sorted[c.inds], y_inds[c.inds]

			countsPS = counts_per_split(c.counts, c_x, c_y, missing_values, sep_nan)
			# print("M PS:", missing_values, "\n", countsPS)
			flat_impurities = criterion_func(criterion_enum,countsPS.reshape((-1,countsPS.shape[2])))
			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))
			# print("IMP:", impurities)

			#Sum of new impurities of left and right side of split
			total_split_impurity = impurities[:,0] + impurities[:,1];
			if(sep_nan): total_split_impurity += impurities[:,2]
			impurity_decrease = c.impurity - (total_split_impurity);
			splits = split_chooser(split_enum,impurity_decrease)

			for j in range(len(splits)):
				split = splits[j]

				if(impurity_decrease[split] <= 0.0):
					nodes[c.parent_node]=TreeNode(LEAF,c.parent_node,List.empty_list(i4_arr),c.counts)
				else:
					mask = c_x[:,split];
					missing = np.argwhere(missing_values[:,1] == split)[:,0]
					# print("missing", split, missing)
					node_l, node_r, node_n = -1, -1, -1
					new_inds_l, new_inds_r, new_inds_n = r_l_n_split(mask,missing)
					
					#New node for left.
					# node_l = new_node(locs,split,new_inds_l, impurities,countsPS,0)
					if (cache_nodes): node_l= akd_get(node_dict,new_inds_l)
					if(node_l == -1):
						node_l = len(nodes)
						if(cache_nodes): akd_insert(node_dict,new_inds_l,node_l)
						ms_impurity_l = impurities[split,0].item()
						if(ms_impurity_l > 0):
							nodes.append(TreeNode(NODE,node_l,List.empty_list(i4_arr),countsPS[split,0]))
							new_contexts.append(SplitContext(new_inds_l,
								ms_impurity_l,countsPS[split,0], node_l))
						else:
							nodes.append(TreeNode(LEAF,node_l,List.empty_list(i4_arr),countsPS[split,0]))
						

					#New node for right.
					# node_r = new_node(locs,split,new_inds_r, impurities,countsPS,1)
					if (cache_nodes): node_r = akd_get(node_dict,new_inds_r)
					if(node_r == -1):
						node_r = len(nodes)
						if(cache_nodes): akd_insert(node_dict,new_inds_r,node_r)
						ms_impurity_r = impurities[split,1].item()
						if(ms_impurity_r > 0):
							nodes.append(TreeNode(NODE,node_r,List.empty_list(i4_arr),countsPS[split,1]))
							new_contexts.append(SplitContext(new_inds_r,
								ms_impurity_r,countsPS[split,1], node_r))
						else:
							nodes.append(TreeNode(LEAF,node_r,List.empty_list(i4_arr),countsPS[split,1]))
						

					#New node for NaN values.
					if(sep_nan and len(new_inds_n) > 0):
						# node_n = new_node(locs,split,new_inds_n, impurities,countsPS,2)
						if (cache_nodes): node_n = akd_get(node_dict,new_inds_n)
						if(node_n == -1):
							node_n = len(nodes)
							if(cache_nodes): akd_insert(node_dict,new_inds_n,node_n)
							ms_impurity_n = impurities[split,2].item()
							if(ms_impurity_n > 0):
								nodes.append(TreeNode(NODE,node_n,List.empty_list(i4_arr),countsPS[split,2]))
								new_contexts.append(SplitContext(new_inds_n,
									ms_impurity_n,countsPS[split,2], node_n))
							else:
								nodes.append(TreeNode(LEAF,node_n,List.empty_list(i4_arr),countsPS[split,2]))
					nodes[c.parent_node].split_data.append(np.array([split, node_l, node_r, node_n],dtype=np.int32))

		contexts = new_contexts

	out = encode_tree(nodes,u_ys)
	return out

@njit(nogil=True,fastmath=True)
def encode_tree(nodes,u_ys):
	'''Takes a list of nodes and encodes them into a 1d-int32 numpy array. 
		The encoding is [length-node_parts, *[nodes...], class_ids] with:
			for each node in nodes : [len_encoding, ttype,index,*[splits...],*counts[:]]
			for each split in split : [feature_index, offset_left, offset_right, offset_nan]
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
		l = 4 + len(node.split_data)*4 + n_classes
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
			out[ind+1] = out_node_slices[sd[1]] if sd[1] != -1 else -1; 
			out[ind+2] = out_node_slices[sd[2]] if sd[2] != -1 else -1; 
			out[ind+3] = out_node_slices[sd[3]] if sd[3] != -1 else -1; 
			ind += 4
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
		splits = slc[4:4+slc[3]*4].reshape(slc[3],4)
	else:
		splits = None
	counts = slc[4+slc[3]*4:]
	
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
					split_on, left, right, nan  = s[0],s[1],s[2],s[3]
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
	y_uvs = _get_y_order(tree)
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



		
@njit(nogil=True,fastmath=True, cache=True, locals={"ONE":i4})
def predict_tree(tree,X,pred_choice_enum,positive_class=0,decode_classes=True):
	'''Predicts the class associated with an unlabelled sample using a fitted 
		decision/ambiguity tree'''
	ONE = 1 
	out = np.empty((X.shape[0],),dtype=np.int64)
	y_uvs = _get_y_order(tree)
	for i in range(len(X)):
		x = X[i]
		nodes = List.empty_list(i4); nodes.append(ONE)
		leafs = List()
		while len(nodes) > 0:
			new_nodes = List()
			for node in nodes:
				ttype, index, splits, counts = _unpack_node(tree,node)
				if(ttype == TreeTypes_NODE):
					for j,s in enumerate(splits):
						split_on, left, right, nan  = s[0],s[1],s[2],s[3]
						if(np.isnan(x[split_on])):
							_n = nan
						elif(x[split_on]):
							_n = right
						else:
							_n = left
						new_nodes.append(_n)
				else:
					leafs.append(counts)
			nodes = new_nodes
		out_i = pred_choice_func(pred_choice_enum,leafs,positive_class)
		if(decode_classes):out_i = y_uvs[out_i]
		out[i] = out_i
	return out


######### Repr/Visualtization #########

def str_tree(tree):
	'''A string representation of a tree usable for the purposes of debugging'''
	
	print(tree)
	l = ["TREE w/ classes: %s"%_get_y_order(tree)]
	node_offset = 1
	while node_offset < tree[0]:
		node_width = tree[node_offset]
		ttype, index, splits, counts = _unpack_node(tree,node_offset)
		if(ttype == TreeTypes_NODE):
			s  = "NODE(%s) : " % (index)
			for split in splits:
				s += "(%s)[L:%s R:%s" % (split[0],_indexOf(tree,split[1]),_indexOf(tree,split[2]))
				s += "] " if(split[3] == -1) else ("NaN:" + _indexOf(tree,split[3]) + "] ")
			l.append(s)
		else:
			s  = "LEAF(%s) : %s" % (index,counts)
			l.append(s)
		node_offset += node_width
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
		def _fit(x,y,missing_values=None):	
			if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
			out =fit_tree(x,y,
					missing_values=missing_values,
					criterion_enum=literally(criterion_enum),
					split_enum=literally(split_enum),
					sep_nan=literally(sep_nan),
					cache_nodes=literally(cache_nodes)
				 )
			return out
		self._fit = _fit

		@njit(cache=True)
		def _predict(tree, X, missing_values=None, positive_class=1):	
			if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
			out =predict_tree(tree,X,
					pred_choice_enum=literally(pred_choice_enum),
					positive_class=positive_class,
					decode_classes=True
				 )
			return out
		self._predict = _predict
		self.tree = None
		
	def fit(self,X,y,missing_values=None):
		self.tree = self._fit(X, y, missing_values)

	def predict(self,X,positive_class=None):
		if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
		if(positive_class is None): positive_class = self.positive_class
		return self._predict(self.tree, X, positive_class)

	def __str__(self):
		return str_tree(self.tree)

	def as_conditions(self,positive_class=None, only_pure_leaves=False):
		if(positive_class is None): positive_class = self.positive_class
		return tree_to_conditions(self.tree, positive_class, only_pure_leaves)


@jit(cache=True)
def test_fit(x,y,missing_values=None):	
	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
	out =fit_tree(x,y,
			missing_values=missing_values,
			criterion_enum=1,
			split_enum=1,
			sep_nan=True
		 )
	return out



@jit(cache=True)
def test_Afit(x,y,missing_values=None):	
	if(missing_values is None): missing_values = np.empty((0,2), dtype=np.int64)
	out =fit_tree(x,y,
			missing_values=missing_values,
			criterion_enum=1,
			split_enum=2,
			cache_nodes=True,
		 )
	return out


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

	# tree = test_fit(data[:,[1,0,2,3,4,5]],labels)
	# treeA = test_Afit(data[:,[1,0,2,3,4,5]],labels)
	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE_pure_majority,positive_class=1))
	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE_pure_majority,positive_class=1))
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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE_pure_majority,positive_class=1))

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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE_pure_majority,positive_class=1))

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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE_pure_majority,positive_class=1))

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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE_pure_majority,positive_class=1))


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





















