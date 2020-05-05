# cython: infer_types=True, language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import numba
from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, unicode_type
import timeit
from sklearn import tree as SKTree
from numbaILP.compile_template import compile_template
from enum import IntEnum
from numba.pycc import CC
from fnvhash import hasharray, AKD#, akd_insert,akd_get

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

		s = 0.0;
		if(total > 0):
			for i in range(counts.shape[1]):
				prob = counts[j,i] / total;
				s += prob * (1.0-prob)

		out[j] = s;
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
def counts_per_split(start_counts, x, y_inds, sep_nan=False):
	counts = np.zeros((x.shape[1],2+sep_nan,len(start_counts)),dtype=np.uint32);
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(sep_nan and np.isnan(x[i,j])):
				counts[j,2,y_inds[i]] += 1;	
			else:
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
		   ('split_on',	 ListType(i4)),
		   ('left',      ListType(i4)),
		   ('right',     ListType(i4)),
		   ('nan',       ListType(i4)),
		   ('counts',    optional(u4[:]))])
class TreeNode(object):
	def __init__(self):
		self.ttype = 0
		self.index = 0
		#For Nodes
		self.split_on = List.empty_list(i4)
		self.left = List.empty_list(i4)
		self.right = List.empty_list(i4)
		self.nan = List.empty_list(i4)
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

@jitclass([#('x', b1[:,:]),
		   #('y', u4[:]),
		   ('inds', u4[:]),
		   ('impurity', f8),
		   ('counts', u4[:]),
		   ('parent_node', TN)])
class SplitContext(object):
	def __init__(self,inds,#,x,y,
		impurity,counts,parent_node):
		# self.x = x
		# self.y = y
		self.inds = inds
		self.impurity = impurity
		self.counts = counts
		self.parent_node = parent_node

@njit(void(TN,i4,i4,i4),nogil=True,fastmath=True)
def assign_node(tn,split_on,left,right):
	tn.ttype = TreeTypes.NODE
	tn.split_on.append(split_on)
	tn.left.append(left)
	tn.right.append(right)


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
def choose_single_max(impurity_decrease):
	return np.asarray([np.argmax(impurity_decrease)])

@njit(nogil=True,fastmath=True,cache=True)
def choose_all_max(impurity_decrease):
	m = np.max(impurity_decrease)
	return np.where(impurity_decrease==m)[0]
	# return np.asarray([np.argmax(impurity_decrease)])

@njit(nogil=True,fastmath=True)
def test_fit(x,y):	
	out =fit_tree(x,y,
			criterion_func=gini,
			split_chooser=choose_single_max,
			sep_nan=True
		 )
	return out

@njit(nogil=True,fastmath=True)
def fit_tree(x,y,criterion_func,split_chooser,sep_nan=False):
	# criterion_func = gini#get_criterion_func(criterion)
	sorted_inds = np.argsort(y)
	x_sorted = x[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);
	impurity = criterion_func(np.expand_dims(counts,0))[0]
	contexts = List()
	tree = Tree()
	contexts.append(SplitContext(np.arange(0,len(x),dtype=np.uint32),#x_sorted,y_inds,
		impurity,counts,new_node(tree)))

	# parent_node = TreeNode()

	while len(contexts) > 0:
		new_contexts = List()

		for i in range(len(contexts)):
			c = contexts[i]
			c_x = x_sorted[c.inds]
			c_y = y_inds[c.inds]
			# print(c_y)

			countsPS = counts_per_split(c.counts,c_x,c_y,sep_nan)
			flat_countsPS = countsPS.reshape((-1,countsPS.shape[2]))
			flat_impurities = criterion_func(flat_countsPS)
			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))

			#Sum of new impurities of left and right side of split
			total_split_impurity = impurities[:,0] + impurities[:,1] 
			if(sep_nan): total_split_impurity += impurities[:,2];
			impurity_decrease = c.impurity - (total_split_impurity);
			splits = split_chooser(impurity_decrease)
			for j in range(len(splits)):
				split = splits[j]

				if(impurity_decrease[split] <= 0):
					assign_leaf(c.parent_node, c.counts)
				else:
					node_l, node_r = new_node(tree), new_node(tree)
					assign_node(c.parent_node,split,node_l.index,node_r.index)

					ms_impurity_l = impurities[split,0].item()
					ms_impurity_r = impurities[split,1].item()

					mask = c_x[:,split];
					if(ms_impurity_l > 0):
						sel = np.argwhere(np.logical_not(mask))[:,0]
						new_contexts.append(SplitContext(c.inds[sel],#c.x[sel], c.y[sel],
							ms_impurity_l,countsPS[split,0], node_l))
					else:
						assign_leaf(node_l, countsPS[split,0])
					
					if(ms_impurity_r > 0):
						sel = np.argwhere(mask)[:,0]
						new_contexts.append(SplitContext(c.inds[sel],#c.x[sel], c.y[sel],
							ms_impurity_r,countsPS[split,1], node_r))
					else:
						assign_leaf(node_r, countsPS[split,1])

					if(sep_nan):
						node_nan = new_node(tree)
						ms_impurity_nan = impurities[split,2].item()
						if(ms_impurity_nan > 0):
							sel = np.argwhere(np.isnan(mask))[:,0]
							new_contexts.append(SplitContext(c.inds[sel],#c.x[sel], c.y[sel],
								ms_impurity_r,countsPS[split,2], node_nan))
						else:
							c.parent_node.nan.append(node_nan.index)


		contexts = new_contexts
	return tree

@njit(nogil=True,fastmath=True)
def test_Afit(x,y):	
	out =fit_Atree(x,y,
			criterion_func=gini,
			split_chooser=choose_all_max
		 )
	return out

# @njit(nogil=True,fastmath=True,inline='always')
# def build_new_node(split_locals,new_inds,split,k):
# 	'''Handle the creation of a new node. For each splitable node this will be run for
# 		the left, right, and NaN children. If the new node would be redundant but comes
# 		from a different set of splits then just retrieve the equivalent node from the
# 		node_dict since all subtrees will be redundant. Otherwise determine if the new 
# 		node will be a leaf or not. If not, push a new context to compute its children.
# 	'''
# 	# new_inds = c.inds[np.argwhere(~mask)[:,0]]
# 	tree,node_dict,new_contexts,countsPS,impurities = split_locals
# 	new_inds_hash = hasharray(new_inds)
# 	if(new_inds_hash not in node_dict):
# 		node_dict[new_inds_hash] = node = new_node(tree)
# 		ms_impurity = impurities[split,k].item()
# 		if(ms_impurity > 0):
# 			new_contexts.append(SplitContext(new_inds,#c.x[sel], c.y[sel],
# 				ms_impurity,countsPS[split,k], node))
# 		else:
# 			# print("L:",countsPS[split,0],split, ms_impurity_l)
# 			assign_leaf(node, countsPS[split,k])
# 	else:
# 		node = node_dict[new_inds_hash]
# 	return node

@njit(nogil=True,fastmath=True)
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

BE, akd_get, akd_includes, akd_insert = AKD(TN)


@njit(nogil=True,fastmath=True,inline='always')
def fit_Atree(x,y,criterion_func,split_chooser,sep_nan=False):
	# criterion_func = gini#get_criterion_func(criterion)
	sorted_inds = np.argsort(y)
	x_sorted = x[sorted_inds]
	counts, u_ys, y_inds = unique_counts(y[sorted_inds]);
	impurity = criterion_func(np.expand_dims(counts,0))[0]
	contexts = List()
	tree = Tree()
	contexts.append(SplitContext(np.arange(0,len(x),dtype=np.uint32),#x_sorted,y_inds,
		impurity,counts,new_node(tree)))

	# parent_node = TreeNode()
	# node_dict = AKD_TN()#Dict.empty(u4,TN)
	node_dict = Dict.empty(u4,BE)#Dict.empty(u4,TN)
	# node_dict = Dict.empty(u4,TN)
	while len(contexts) > 0:
		new_contexts = List()
		for i in range(len(contexts)):
			c = contexts[i]
			c_x, c_y = x_sorted[c.inds], y_inds[c.inds]
			# print(c_y)

			countsPS = counts_per_split(c.counts,c_x,c_y,sep_nan)
			flat_countsPS = countsPS.reshape((-1,countsPS.shape[2]))
			flat_impurities = criterion_func(flat_countsPS)
			impurities = flat_impurities.reshape((countsPS.shape[0],countsPS.shape[1]))
			# print(impurities)
			#Sum of new impurities of left and right side of split
			total_split_impurity = impurities[:,0] + impurities[:,1];
			if(sep_nan): total_split_impurity += impurities[:,2]
			impurity_decrease = c.impurity - (total_split_impurity);
			# print(c.impurity,criterion_func(np.expand_dims(c.counts,0)))
			# print(impurity_decrease)
			splits = split_chooser(impurity_decrease)

			# split_locals = (tree,node_dict,new_contexts,countsPS,impurities)
			for j in range(len(splits)):
				split = splits[j]

				if(impurity_decrease[split] <= 0.0):
					assign_leaf(c.parent_node, c.counts)
				else:
					mask = c_x[:,split];

					new_inds_l, new_inds_r, new_inds_n = r_l_n_split(mask)
					
					#New node for left.
					# new_inds_lt = hasharray(new_inds_l)
					# if(new_inds_lt not in node_dict):
					# 	node_dict[new_inds_lt] = node_l = new_node(tree)
					node_l = akd_get(node_dict,new_inds_l)
					if(node_l is None):
						node_l = new_node(tree)
						akd_insert(node_dict,new_inds_l,node_l)
						ms_impurity_l = impurities[split,0].item()
						if(ms_impurity_l > 0):
							new_contexts.append(SplitContext(new_inds_l,#c.x[sel], c.y[sel],
								ms_impurity_l,countsPS[split,0], node_l))
						else:
							# print("L:",countsPS[split,0],split, ms_impurity_l)
							assign_leaf(node_l, countsPS[split,0])
					# else:
					# 	node_l = node_dict[new_inds_lt]


					#New node for right.
					# new_inds_rt = hasharray(new_inds_r)
					# if(new_inds_rt not in node_dict):
					# 	node_dict[new_inds_rt] = node_r = new_node(tree)
					node_r = akd_get(node_dict,new_inds_r)
					if(node_r is None):
						node_r = new_node(tree)
						akd_insert(node_dict,new_inds_r,node_r)
						ms_impurity_r = impurities[split,1].item()
						if(ms_impurity_r > 0):
							new_contexts.append(SplitContext(new_inds_r,#c.x[sel], c.y[sel],
								ms_impurity_r,countsPS[split,1], node_r))
						else:
							# print("R:",countsPS[split,1])
							assign_leaf(node_r, countsPS[split,1])
					# else:
					# 	node_r = node_dict[new_inds_rt]


					assign_node(c.parent_node, split, node_l.index, node_r.index)
					#New node for NaN values.
					# if(sep_nan):
					# 	new_inds_nt = hasharray(new_inds_n)
					# 	if(new_inds_nt not in node_dict):
					# 		node_dict[new_inds_nt] = node_n = new_node(tree)
					# 		ms_impurity_n = impurities[split,2].item()
					# 		if(ms_impurity_n > 0):
					# 			new_contexts.append(SplitContext(new_inds_n,#c.x[sel], c.y[sel],
					# 				ms_impurity_n,countsPS[split,2], node_n))
					# 		else:
					# 			assign_leaf(node_n, countsPS[split,2])
					# 	else:
					# 		node_n = node_dict[new_inds_nt]
					# 	c.parent_node.nan.append(node_n)

		contexts = new_contexts
	return tree

def str_tree(tree):
	l = []
	for i in range(len(tree.nodes)):
		tn = tree.nodes[i]
		if(tn.ttype == TreeTypes.NODE):
			if(len(tn.nan) == 0):
				l.append("%s : %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right))
			else:
				l.append("%s : %s %s %s %s" % (tn.index, tn.split_on, tn.left, tn.right, tn.nan))
		else:
			l.append("%s : %s" % (tn.index, tn.counts))
	return "\n".join(l)

def print_tree(tree):
	print(str_tree(tree))
	# for i in range(len(tree.nodes)):
	# 	tn = tree.nodes[i]
	# 	if(tn.ttype == TreeTypes.NODE):
	# 		print(tn.index,": ",tn.split_on, tn.left, tn.right)
	# 	else:
	# 		print(tn.index,": ",tn.counts)


# def purest_leaf(leaf_counts):

# @njit(nogil=True,fastmath=True)
def choose_majority_general(leaf_counts,positive_class):
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return True
	return False

@njit(nogil=True,fastmath=True)
def choose_pure_majority_general(leaf_counts,positive_class):
	pure_counts = List()
	# for i in range(len(leaf_counts)):
	for count in leaf_counts:
		# count = leaf_counts[i]
		# if(np.sum(count > 0) == 1):
		print("count",count,count.dtype)
		if(np.count_nonzero(count) == 1):
			pure_counts.append(count)
	leaf_counts = pure_counts if len(pure_counts) > 0 else leaf_counts
	for i,count in enumerate(leaf_counts):
		pred = np.argmax(count)
		if(pred == positive_class):
			return True
	return False


@njit(nogil=True,fastmath=True)
def predict_tree(tree,X,choice_func,positive_class=0):
	out = np.empty((X.shape[0]))
	
	for i in range(len(X)):
		x = X[i]
		nodes = List(); nodes.append(tree.nodes[0])
		leafs = List()
		while len(nodes) > 0:
			new_nodes = List()
			for node in nodes:
				if(node.ttype == TreeTypes.NODE):
					for j,s in enumerate(node.split_on):
						_n = node.right[j] if x[s] else node.left[j]
						# for _n in sub_nodes:
						# if(tree.nodes[_n].ttype == TreeTypes.LEAF):
						# 	print(s,_n, "["+",".join([str(_x) for _x in tree.nodes[_n].counts])+"]" )
						# else:
						# 	print(s,_n,)
						new_nodes.append(tree.nodes[_n])
				else:
					# out[i] = np.argmax(node.counts)
					### TODO: Need to copy to unoptionalize the type: remove [:] if #4382 ever fixed
					leafs.append(node.counts[:])
			nodes = new_nodes
		# print(str(i)+":",["["+",".join([str(_x) for _x in l])+"]" for l in leafs])
		out[i] = choice_func(leafs,positive_class)
		
	return out
		
class TreeClassifier(object):
	def __init__(self, tree_type='ambiguity',
					  criterion_func='gini',
					  split_chooser='choose_all_max',
					  choice_func='choose_pure_majority_general',
					  positive_class=1):
		l = globals()
		criterion_func = l[criterion_func]
		split_chooser = l[split_chooser]
		choice_func = l[choice_func]
		positive_class = positive_class
		
		ft = fit_Atree if tree_type == 'ambiguity' else fit_tree
		@njit(nogil=True,fastmath=True)
		def _fit(X,y):
			return ft(X,y,criterion_func,split_chooser)
		self._fit = _fit

		@njit(nogil=True,fastmath=True)
		def _predict(tree,X):
			return predict_tree(tree,X,choice_func,positive_class)
		self._predict = _predict

		self.tree = None
		
	def fit(self,X,y):
		self.tree = self._fit(X,y)

	def predict(self,X):
		return self._predict(self.tree,X)

	def __str__(self):
		return str_tree(self.tree)

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
	clf = SKTree.DecisionTreeClassifier()
	# my_bdt = ILPTree()
	my_AT = TreeClassifier()


	# nb_fit = my_bdt.nb_ilp_tree.fit

	##Compiled 
	# cc = CC("my_module")
	# compile_template(fit_tree,{'criterion_func': gini},cc,'TR(b1[:,:],u4[:])',globals())
	# cc.compile()
	# from my_module import fit_tree_gini	
	# def c_bdt():
	# 	fit_tree_gini(data,labels)
	###

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
	
	N = 10000
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

	labels = np.asarray([1,1,1,2,2,2],np.int32);
	data = data[:,[1,0,2,3,4,5]]

	# tree = test_fit(data[:,[1,0,2,3,4,5]],labels)
	# treeA = test_Afit(data[:,[1,0,2,3,4,5]],labels)
	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,choose_pure_majority_general,positive_class=1))
	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,choose_pure_majority_general,positive_class=1))
	my_AT.fit(data,labels)
	print("MY_AT",my_AT.predict(data))
	print("MY_AT",my_AT)

	data = np.asarray([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0], #1
	[1,0], #1
	[0,1], #1
	[1,1], #2
	],np.bool);

	labels = np.asarray([1,1,1,2],np.int32);

	tree = test_fit(data,labels)
	treeA = test_Afit(data,labels)
	print("___")
	print_tree(tree)
	print("PREDICT DT",predict_tree(tree,data,choose_pure_majority_general,positive_class=1))

	print("___")
	print_tree(treeA)
	print("PREDICT AT",predict_tree(treeA,data,choose_pure_majority_general,positive_class=1))


	clf = SKTree.DecisionTreeClassifier()
	clf.fit(data,labels)
	print(clf.predict(data[[-1]])	)


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






