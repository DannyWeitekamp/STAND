# cython: infer_types=True, language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import numba
from numba import types, njit, guvectorize,vectorize,prange, jit
from numba.experimental import jitclass
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
from fnvhash import hasharray, AKD#, akd_insert,akd_get





#########  Impurity Functions #######
class CRITERION(IntEnum):
	gini = 1
	return_zero = 2

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


@njit(cache=True, inline='always')
def criterion_func(func_enum,counts):
	if(func_enum == CRITERION.gini):
		return gini(counts)
	elif(func_enum == CRITERION.return_zero):
		return return_zero(counts)
	return gini(counts)


######### Split Choosers ##########
class SPLIT_CHOICE(IntEnum):
	choose_single_max = 1
	choose_all_max = 2

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


@njit(cache=True,inline='always')
def split_chooser(func_enum,impurity_decrease):
	if(func_enum == SPLIT_CHOICE.choose_single_max):
		return choose_single_max(impurity_decrease)
	elif(func_enum == SPLIT_CHOICE.choose_all_max):
		return choose_all_max(impurity_decrease)
	return choose_single_max(impurity_decrease)

######### Prediction Choice Functions #########
class PRED_CHOICE(IntEnum):
	choose_majority_general = 1
	choose_pure_majority_general = 2


@njit(nogil=True,fastmath=True,cache=True)
def choose_majority_general(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority of leaves.''' 
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return True
	return False

@njit(nogil=True,fastmath=True,cache=True)
def choose_pure_majority_general(leaf_counts,positive_class):
	''' If multiple leaves on predict (i.e. ambiguity tree), choose 
		the class predicted by the majority pure of leaves.''' 
	pure_counts = List()
	for count in leaf_counts:
		if(np.count_nonzero(count) == 1):
			pure_counts.append(count)
	leaf_counts = pure_counts if len(pure_counts) > 0 else leaf_counts
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return True
	return False

@njit(nogil=True,fastmath=True,cache=True)
def pred_choice_func(func_enum,leaf_counts,positive_class):
	if(func_enum == PRED_CHOICE.choose_majority_general):
		return choose_majority_general(leaf_counts,positive_class)
	elif(func_enum == PRED_CHOICE.choose_pure_majority_general):
		return choose_pure_majority_general(leaf_counts,positive_class)
	return choose_majority_general(leaf_counts,positive_class)





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

class TreeTypes(IntEnum):
	NODE = 1
	LEAF = 2

######### Struct Definitions #########

# TN = NamedTuple([i8,i8,ListType(i8[::1]),u8[::1]],TreeNode)
# @jitclass([('ttype',	   i8),
# 		   ('index',	   i8),
# 		   ('split_on',	 ListType(i8)),
# 		   ('left',      ListType(i8)),
# 		   ('right',     ListType(i8)),
# 		   ('nan',       ListType(i8)),
# 		   ('counts',    optional(u8[:]))])
# class TreeNode(object):
# 	'''A particular node in the tree
# 		ttype -- Indicates if it is a leaf or node
# 		index -- The location of the node in the list of all nodes
# 		split_on -- If is a non-leaf node, the set of splits made on this node
# 			can be more than one in the case of ambiguity tree
# 		left -- For each split in split_on the index of the node to the left
# 		right -- For each split in split_on the index of the node to the right
# 		nan -- For each split in split_on the index of the node in the nan slot
# 		counts -- If is a leaf node the number of samples of each class falling in it
# 	'''

# 	def __init__(self):
# 		self.ttype = 0
# 		self.index = 0
# 		#For Nodes
# 		self.split_on = List.empty_list(i8)
# 		self.left = List.empty_list(i8)
# 		self.right = List.empty_list(i8)
# 		self.nan = List.empty_list(i8)
# 		#For Leaves
# 		self.counts = None

# TN = TreeNode.class_type.instance_type

# Tree = namedtuple("Tree",['nodes'])
# TR = NamedTuple([DictType(i8,TN)],Tree)

# @jitclass([('nodes',DictType(i8,TN))])
# class Tree(object):
# 	'''A list of nodes'''
# 	def __init__(self):
# 		self.nodes = Dict.empty(i8,TN)#List.empty_list(TN)

# TR = Tree.class_type.instance_type

# @jitclass([('inds', u4[::1]),
# 		   ('impurity', f8),
# 		   ('counts', u4[::1]),
# 		   ('parent_node', i4)])
# class SplitContext(object):
# 	''' An object holding relevant local variables of the tree after a split.
# 		This is used to avoid using recursion.
# 		inds -- A list of indicies of samples which fall in the present branch of the tree.
# 		impurity -- The impurity of this branch of the tree.
# 		counts -- The number of samples of each class.
# 		parent node -- The node from which this branch was produced.
# 	'''
# 	def __init__(self,inds,
# 		impurity,counts,parent_node):
# 		self.inds = inds
# 		self.impurity = impurity
# 		self.counts = counts
# 		self.parent_node = parent_node

######### Utility Functions for Fit/Predict  #########

@njit(nogil=True,fastmath=True,cache=True)
def counts_per_split(start_counts, x, y_inds, sep_nan=False):
	''' 
		Determines the number of elements of each class that would be in the resulting
		left, right and nan nodes if a split was made at each possible feature.
	'''
	counts = np.zeros((x.shape[1],2+sep_nan,len(start_counts)),dtype=np.uint32);
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(sep_nan and np.isnan(x[i,j])):
				counts[j,2,y_inds[i]] += 1;	
			else:
				if(x[i,j]):
					counts[j,1,y_inds[i]] += 1;	
				else:
					counts[j,0,y_inds[i]] += 1;	
	return counts;



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
	# c = np.empty(len(counts), dtype=np.uint64)#np.asarray(counts,dtype=np.uint64)
	# for i,v in enumerate(c):
	# 	c[i] = v
	u = np.asarray(uniques,dtype=np.int32)
	return c, u, inds




# @njit(void(TR,i8,i8,u8[::1],i8,i8,optional(i8)),nogil=True,fastmath=True)
# def assign_node(tree,tn_ind,split_on,counts,left,right,nan=None):
# 	'''Sets as NODE type and fills in content of node'''
# 	_nan = np.array(-1,dtype=np.int64).item() #if(nan is None) else nan
# 	# split_data = SplitData(split_on,left,right,_nan)
# 	split_data_list = tree.nodes[tn_ind].split_data
# 	split_data_list.append(np.array([split_on,left,right,_nan],dtype=np.int64))#split_data)
# 	# tn = TreeNode(TreeTypes.NODE,tn_ind,split_data,None)
# 	# tn.ttype = TreeTypes.NODE
# 	# tn.split_on.append(split_on)
# 	# tn.left.append(left)
# 	# tn.right.append(right)
# 	# if(nan is not None): tn.nan.append(nan)
# 	tree.nodes[tn_ind] = TreeNode(2,tn_ind,split_data_list,counts)


# # @njit(void(TN,u8[:]),nogil=True,fastmath=True,cache=True)
# @njit(void(TR,i8,u8[::1]),nogil=True,fastmath=True,cache=True)
# def assign_leaf(tree,tn_ind,counts):
# 	'''Sets as LEAF type Fills in counts'''
# 	# tn = TreeNode()
# 	# tn.ttype = TreeTypes.LEAF
# 	# tn.counts = counts
# 	tree.nodes[tn_ind] = TreeNode(1,tn_ind,tree.nodes[tn_ind].split_data,counts)

# # @njit(TN(TR),nogil=True,fastmath=True)
# @njit(i8(TR,u8[::1]),nogil=True,fastmath=True)
# def new_node(tree,counts):
# 	return 1
	'''Instantiates a new node, of undetermined type'''
	# index = len(tree.nodes)
	# tn = TreeNode(0,np.array(len(tree.nodes),dtype=np.int64).item(),np.empty(4,dtype=np.int64),counts)
	
	# tree.nodes.append(tn)
	# tree.nodes.append(tn)
	# tn.index = index
	# return index


@njit(nogil=True,fastmath=True,cache=True)
def r_l_n_split(x,sep_nan=False):
	'''Similar to argwhere applied 3 times each for 0,1 and nan, but does all
	    three at once.'''
	nl,nr,nn = 0,0,0
	l = np.empty(x.shape,np.uint32)
	r = np.empty(x.shape,np.uint32)
	n = np.empty(x.shape,np.uint32)
	
	for i in range(len(x)):
		x_i = x[i]
		if(sep_nan and x_i == 255):
			n[nn] = i
			nn += 1
		else:
			if(x[i]):
				r[nr] = i
				nr += 1
			else:
				l[nl] = i
				nl += 1

	# return np.array(l), np.array(r), np.array(n)
	return l[:nl], r[:nr], n[:nn]

BE = Tuple([u1[::1],i4])

@njit(nogil=True,fastmath=True)
def akd_insert(akd,_arr,item,h=None):
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
    arr = _arr.view(np.uint8)
    if(h is None): h = hasharray(arr) 
    if(h in akd):
	    for elem in akd[h]:
	        if(len(elem[0]) == len(arr) and
	            (elem[0] == arr).all()): 
	            return elem[1]
    return -1

######### Fit #########

SplitData = namedtuple("SplitData",['split_on','left','right','nan'])
NBSplitData = NamedUniTuple(i4,4,SplitData)


TreeNode = namedtuple("TreeNode",['ttype','index','split_data','counts'])
TN = NamedTuple([i4,i4,ListType(i4[:]),u4[::1]],TreeNode)

SplitContext = namedtuple("SplitContext",['inds','impurity','counts','parent_node'])
SC = NamedTuple([u4[::1],f8,u4[::1],i4],SplitContext)

# jitclass([('inds', u4[::1]),
# 		   ('impurity', f8),
# 		   ('counts', u4[::1]),
# 		   ('parent_node', i4)])

# Tree = namedtuple("Tree",['nodes'])
# TR = NamedTuple([DictType(i4,TN)],Tree)

@njit
def new_node(indx,counts):
	return TreeNode(1,indx,List.empty_list(NBSplitData),counts)

@njit
def new_leaf(indx,counts):
	return TreeNode(2,indx,List.empty_list(NBSplitData),counts)

i4_arr = i4[:]
BE_List = ListType(BE)

# @njit
# def _resolve_criterion(f_name):
# 	if(f_name == "gini"):
# 		return gini
# 	return gini

# @njit
# def _resolve_split_chooser(f_name):
# 	if(f_name == "choose_all_max"):
# 		return choose_all_max
# 	elif(f_name == "choose_single_max"):
# 		return choose_single_max
# 	return choose_single_max






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
def fit_tree(x,y,criterion_enum,split_enum,sep_nan=False, cache_nodes=False):
	'''Fits a decision/ambiguity tree'''
	# _criterion_func, _split_chooser = _resolve_criterion(criterion_func), _resolve_split_chooser(split_chooser)
	#ENUMS
	ZERO,NODE, LEAF = 0,1, 2
	# criterion_func = gini#get_criterion_func(criterion)
	sorted_inds = np.argsort(y)
	x_sorted = x[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);
	impurity = criterion_func(criterion_enum,np.expand_dims(counts,0))[0]

	contexts = List.empty_list(SC)
	contexts.append(SplitContext(np.arange(0,len(x),dtype=np.uint32),impurity,counts,ZERO))

	node_dict = Dict.empty(u4,BE_List)
	nodes = List.empty_list(TN)
	nodes.append(TreeNode(NODE,ZERO,List.empty_list(i4_arr),counts))
	# n_nodes = 1
	

	while len(contexts) > 0:
		new_contexts = List.empty_list(SC)
		# locs = (node_dict,nodes,new_contexts,cache_nodes)
		for i in range(len(contexts)):
			c = contexts[i]
			c_x, c_y = x_sorted[c.inds], y_inds[c.inds]

			countsPS = counts_per_split(c.counts,c_x,c_y,sep_nan)
			flat_impurities = criterion_func(criterion_enum,countsPS.reshape((-1,countsPS.shape[2])))
			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))

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
					node_l, node_r, node_n = -1, -1, -1
					new_inds_l, new_inds_r, new_inds_n = r_l_n_split(mask)
					
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

	out = encode_tree(nodes)
	return out

@njit(nogil=True,fastmath=True)
def encode_tree(nodes):
	n_classes = len(nodes[0].counts)
	out_node_slices = np.empty((len(nodes)+1,),dtype=np.int32)
	out_node_slices[0] = 0
	offset = 0 
	for i,node in enumerate(nodes):
		l = 4 + len(node.split_data)*4 + n_classes
		offset += l 
		out_node_slices[i+1] = offset
		# out_node_slices[i,1] = offset+l
	# print(n_classes,out_node_slices)
	out = np.empty((offset,),dtype=np.int32)
	for i,node in enumerate(nodes):
		# print(i,":",node.ttype,node.index,node.split_data,node.counts)
		ind = out_node_slices[i]

		out[ind+0] = out_node_slices[i+1]-out_node_slices[i]
		out[ind+1] = node.ttype 
		out[ind+2] = node.index
		out[ind+3] = len(node.split_data)
		ind += 4
		# print(node.split_data)
		for sd in node.split_data:

			out[ind+0] = sd[0]; 
			out[ind+1] = out_node_slices[sd[1]] if sd[1] != -1 else -1; 
			out[ind+2] = out_node_slices[sd[2]] if sd[2] != -1 else -1; 
			out[ind+3] = out_node_slices[sd[3]] if sd[3] != -1 else -1; 
			# print(sd,out[ind:ind+4])
			ind += 4
		# print(ind,out_node_slices[i+1])
		out[ind:out_node_slices[i+1]] = node.counts

	return out

@njit(cache=True,inline='always')
def _unpack_node(tree,node_offset):
	l  = tree[node_offset]
	slc = tree[node_offset:node_offset+l]
	# print(slc)
	ttype = slc[1]
	index = slc[2]
	if(ttype == TreeTypes.NODE):
		# print(ttype,slc[3],slc[4:4+slc[3]*4].shape,(slc[3],4))
		splits = slc[4:4+slc[3]*4].reshape(slc[3],4)
		# for i, split in enumerate(splits):
		# 	for j in (1,2,3):
		# 		splits[i,j] = _indexOf(splits[i,j])
	else:
		splits = None
	counts = slc[4+slc[3]*4:]
	
	return ttype, index, splits, counts

@njit(cache=True,inline='always')
def _indexOf(tree,node_offset):
	return tree[node_offset+2]

		
# def decode_tree(tree):


		
	# tree = Tree(nodes)
	# return tree
	# return nodes
  

######### Predict #########

@njit(nogil=True,fastmath=True, cache=True, locals={"ZERO":i4})
def predict_tree(tree,X,pred_choice_enum,positive_class=0):
	'''Predicts the class associated with an unlabelled sample using a fitted 
		decision/ambiguity tree'''
	ZERO = 0 
	out = np.empty((X.shape[0]))
	
	for i in range(len(X)):
		x = X[i]
		nodes = List.empty_list(i4); nodes.append(ZERO)
		leafs = List()
		while len(nodes) > 0:
			new_nodes = List()
			for node in nodes:
				ttype, index, splits, counts = _unpack_node(tree,node)
				if(ttype == TreeTypes.NODE):
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
					### TODO: Need to copy to unoptionalize the type: remove [:] if #4382 ever fixed
					leafs.append(counts)
			nodes = new_nodes
		out[i] = pred_choice_func(pred_choice_enum,leafs,positive_class)
		
	return out


######### Repr/Visualtization #########

def str_tree(tree):
	node_offset = 0
	print(tree)
	l = []
	while node_offset < len(tree):
		node_width = tree[node_offset]
		ttype, index, splits, counts = _unpack_node(tree,node_offset)
		if(ttype == TreeTypes.NODE):
			s  = "NODE(%s) : " % (index)
			for split in splits:
				s += "(%s)[L:%s R:%s" % (_indexOf(tree,split[0]),_indexOf(tree,split[1]),_indexOf(tree,split[2]))
				s += "] " if(split[3] == -1) else ("NaN:" + split[3] + "] ")
			l.append(s)
		else:
			s  = "LEAF(%s) : %s" % (index,counts)
			l.append(s)
		print(tree[node_offset:node_offset+node_width])
		node_offset += node_width
	return "\n".join(l)
	# for i in range(len(tree.nodes)):
	# 	tn = tree.nodes[i]
	# 	if(tn.ttype == TreeTypes.NODE):
	# 		if(len(tn.nan) == 0):
	# 			l.append("%s : %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right))
	# 		else:
	# 			l.append("%s : %s %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right, tn.nan))
	# 	else:
	# 		l.append("%s : %s" % (tn.index, tn.counts))
	# return "\n".join(l)

# def str_tree(tree):
# 	l = []
# 	for i in range(len(tree.nodes)):
# 		tn = tree.nodes[i]
# 		if(tn.ttype == TreeTypes.NODE):
# 			if(len(tn.nan) == 0):
# 				l.append("%s : %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right))
# 			else:
# 				l.append("%s : %s %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right, tn.nan))
# 		else:
# 			l.append("%s : %s" % (tn.index, tn.counts))
# 	return "\n".join(l)

def print_tree(tree):
	print(str_tree(tree))
		
class TreeClassifier(object):
	def __init__(self, 
					  criterion_func='gini',
					  split_chooser='choose_all_max',
					  pred_choice_func='choose_pure_majority_general',
					  positive_class=1):

		l = globals()
		criterion_func = l[criterion_func]
		split_chooser = l[split_chooser]
		pred_choice_func = l[pred_choice_func]
		positive_class = positive_class
		
		ft = fit_tree
		@njit(nogil=True,fastmath=True)
		def _fit(X,y):
			return ft(X,y,criterion_func,split_chooser)
		self._fit = _fit

		@njit(nogil=True,fastmath=True)
		def _predict(tree,X):
			return predict_tree(tree,X,pred_choice_func,positive_class)
		self._predict = _predict

		self.tree = None
		
	def fit(self,X,y):
		self.tree = self._fit(X,y)

	def predict(self,X):
		return self._predict(self.tree,X)

	def __str__(self):
		return str_tree(self.tree)


@jit(cache=True)
def test_fit(x,y):	
	out =fit_tree(x,y,
			criterion_enum=CRITERION.gini,
			split_enum=SPLIT_CHOICE.choose_single_max, #choose_single_max,
			sep_nan=True
		 )
	return out



@jit(cache=True)
def test_Afit(x,y):	
	out =fit_tree(x,y,
			criterion_enum=CRITERION.gini,
			split_enum=SPLIT_CHOICE.choose_all_max,#choose_all_max,
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

	print("d_tree:   ", time_ms(bdt))
	print("a_tree:   ", time_ms(At))
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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE.choose_pure_majority_general,positive_class=1))
	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE.choose_pure_majority_general,positive_class=1))
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
	print("PREDICT DT",predict_tree(tree,data,PRED_CHOICE.choose_pure_majority_general,positive_class=1))

	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,PRED_CHOICE.choose_pure_majority_general,positive_class=1))


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





















