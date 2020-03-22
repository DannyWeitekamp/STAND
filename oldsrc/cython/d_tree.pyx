# cython: infer_types=True, language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from cpython cimport array
import timeit

from sklearn import tree

# from libcpp.map cimport map as cpp_map
# from libcpp cimport bool
# from cython import uint


# from numpy cimport ndarray
# from 
# cdef nfloat = np.
ctypedef np.ndarray ndarray 
ctypedef unsigned char bool
ctypedef unsigned int uint
ctypedef unsigned int uint32
ctypedef unsigned short uint16
ctypedef unsigned long uint64

ctypedef double (*e_func)(uint[::1] counts) nogil

ctypedef fused common_types:
	int
	uint
	long
	uint64
	double
	float

# cdef common_types[:]



cdef e_func get_entropy_func(str name):
	name = name.lower()
	if(name == 'gini'):
		return gini
	return NULL;

cdef double gini(uint[::1] counts) nogil:
	cdef double total = 1e-10;
	cdef int i;
	for i in range(counts.shape[0]):
		total += counts[i];

	# cdef double[:] sq_probs = malloc(len(counts)*sizeof(sizeof(double)))
	cdef double sum_sqr = 0.0;
	cdef double prob;
	for i in range(counts.shape[0]):
		prob = (<double> counts[i]) / total;
		sum_sqr += prob * prob


	# cdef double totals = np.sum(counts,dtype=np.double) + 1e-10

# 	# for i in range(len(counts)):
# # 
# 	prob = counts / totals;
# 	cdef double out = (1.0-np.sum(np.square(prob)))
	return 1.0-sum_sqr;

cdef calc_entropy(uint[:,::1] counts,e_func entropy_func):
	cdef np.ndarray[double,ndim=1] out = np.empty(counts.shape[0], dtype=np.double)
	cdef double[:] out_v = out
	cdef int i;
	# for i in prange(counts.shape[0],nogil=True):
	for i in prange(counts.shape[0],nogil=True):
		out_v[i] = entropy_func(counts[i])
	return out




cdef counts_per_split(uint[::1] start_counts, bool[:,::1] x, uint[::1] y_inds):
	# //(n, s)
	# // xarray<size_t> size = 
	# // cout << start_counts << std::flush;
	# // cout << start_counts.size() << std::flush;
	# cdef uint[:,:,:] counts = np.zeros((x.shape[1],2,len(start_counts)),dtype=np.uint32);
	cdef uint[:,:,:] counts = np.zeros((x.shape[1],2,len(start_counts)),dtype=np.uint32);
	# // auto zeros = (x == 0)
	
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			# // cout << i << ", " << j << ", " << y_inds[i] << std::endl <<std::flush;	
			if(x[i,j]):
				counts[j,1,y_inds[i]] += 1;	
			else:
				counts[j,0,y_inds[i]] += 1;	
	return counts;


# cdef masked_select(common_types[:] inp, bool[:] mask):
# 	uint total = np.sum(mask)
# 	common_types[:] out = np.empty(inp.shape[0])


cdef void split_tree(bool[:,::1] x, uint[::1] y_inds, 
				double entropy,
			    e_func entropy_func,
			    uint[::1] start_counts):
	# pass
	# print("whoop")
	# print(entropy_func(np.asarray([[1,2,3]],dtype=np.uint32)))
	# print(entropy)
	# print(entropy_func)
	# print(start_counts)

	# // cout << "C" << std::endl << std::flush;
	
	cdef uint[:,:,::1] counts = counts_per_split(start_counts,x,y_inds)
	cdef uint[:,::1] counts_l = counts[:,0,:]
	cdef uint[:,::1] counts_r = counts[:,1,:]

	cdef uint[:,::1] flat_counts = np.asarray(counts).reshape((-1,counts.shape[2]))

	# print(counts_l)
	# print(counts_r)
	entropies = calc_entropy(flat_counts,entropy_func).reshape((counts.shape[0],counts.shape[1]))
	entropy_l = entropies[:,0]
	entropy_r = entropies[:,1]
	entropy_lr = entropy_l + entropy_r;

	# print(entropy_l)
	# print(entropy_r)
	utility = entropy - (entropy_lr);
	cdef uint max_split = np.argmax(utility);

	cdef uint[:] max_count
	if(utility[max_split] <= 0):
		max_count = counts[max_split,0,:];
		# print(np.asarray(max_count))
		# vector<uint>* leaf_data = new vector<uint>(max_count.size());
		# cout << "MEEP: " << max_count;
		# for(uint i=0;i < max_count.size();i++){
			# (*leaf_data)[i] = max_count[i];
		# }
		# Leaf *leaf = new Leaf();
		# leaf->data = leaf_data; 
		# return q(TreeType *) leaf;
		return
	# }
	
	# // cout << utility << std::endl;
	# print(counts)
	# print(utility)
	# print(max_split)
	
	# // void* out;
	
	# Tree *left_Tree;	
	# Tree *right_Tree;	
	cdef bool[:] mask;
	cdef np.ndarray[long,ndim=1] sel;
	if(entropy_lr[max_split] > 0):
	# 	cout << ":";

		mask = x[:,max_split];
	# 	// cout << "MASK" << mask << std::endl;
		if(entropy_l[max_split] > 0):
	# 		// auto r = xt::arange(x.shape()[0]);
			sel = np.argwhere(np.logical_not(mask))[:,0]
			split_tree(np.asarray(x)[sel],np.asarray(y_inds)[sel],
				entropy_l[max_split],gini,counts_l[max_split])

			# print("L",np.asarray(sel))
			# auto sel_l = view(xt::from_indices(xt::argwhere(!mask)),all(),0);
	# 		// cout << "SEL_L" << sel_l <<std::endl;
	# 		// cout << "X" << view(x,keep(sel_l),all()) << std::endl;
	# 		cout << " L(";
			# left_Tree = (Tree *)split_tree(view(x,keep(sel_l),all()),
			# 			 		   view(y_inds,keep(sel_l),all()),
			# 			 entropy_l[max_split], gini,counts_l[max_split]);
	# 		cout << ")";
		else:
			pass
	# 		left_Tree = new Tree();
		# }

		if(entropy_r[max_split] > 0):
	# 		auto sel_r = view(xt::from_indices(xt::argwhere(mask)),all(),0);
	# 		// cout << "SEL_R" << sel_r << std::endl;
			sel = np.argwhere(mask)[:,0]
			# print("R",np.asarray(sel))
			# print(np.asarray(x)[sel])
			# print(np.asarray(y_inds)[sel])
	# 		cout << " R(";
			split_tree(np.asarray(x)[sel],np.asarray(y_inds)[sel],
				entropy_r[max_split],gini,counts_r[max_split])
	# 		right_Tree = (Tree *)split_tree(view(x,keep(sel_r),all()),
	# 					 		    view(y_inds,keep(sel_r),all()),
	# 					 entropy_r[max_split], gini,counts_r[max_split]);
	# 		cout << ")";
		else:
			pass
	# 		right_Tree = new Tree();
	# 	}
	# 	// if()
	# }else{
	# 	left_Tree = new Tree();
	# 	right_Tree = new Tree();
	# }
	# Tree *out = new Tree();
	# out->split_on = max_split;
	# out->left = left_Tree;
	# out->right = right_Tree;
	# return (TreeType *)out;


# def binary_decision_tree(np.ndarray[bool, ndim=2] x, np.ndarray[int, ndim=1] y,str entropy_func_name='gini'):
	# c_binary_decision_tree(x,y,entropy_func_name)	
# (np.ndarray[uint,ndim=1],np.ndarray[int,ndim=1],np.ndarray[uint,ndim=1])
cdef unique_counts(int[::1] inp):
	# vector<uint> counts;
	# vector<int> uniques;
	counts = [];
	uniques = [];
	cdef ndarray[uint,ndim=1] inds = np.zeros(len(inp),np.uint32);
	ind=0;
	last = 0;
	cdef uint i;
	for i in range(1,len(inp)):
		if(inp[i-1] != inp[i]):
			counts.append(i-last);
			uniques.append(inp[i-1]);
			last = i;
			ind += 1;
		inds[i] = ind;
	counts.append((i+1)-last);
	uniques.append(inp[i]);

	cdef ndarray[uint,ndim=1] c = np.asarray(counts,dtype=np.uint32)
	cdef ndarray[int,ndim=1] u = np.asarray(uniques,dtype=np.int32)
	# return std::make_tuple(xt::adapt(counts),xt::adapt(uniques),inds);
	return c, u, inds

cpdef binary_decision_tree(np.ndarray[bool, ndim=2] x, np.ndarray[int, ndim=1] y,str entropy_func_name='gini'):
	
	entropy_func = get_entropy_func(entropy_func_name)
	if(entropy_func == NULL):
		raise 0

	# print(np.asarray(x,np.uint))
	# print(np.asarray(y))
	sorted_inds = np.argsort(y).astype(np.int32)
	# print(sorted_inds)
	# print(sorted_inds.dtype)
	# print(int)
	x_sorted = x[sorted_inds]
	y_sorted = y[sorted_inds]
	
	# print(np.asarray(x_sorted,np.uint))
	# print(y_sorted)
	counts, u_ys, y_inds = unique_counts(y_sorted);
	# u_ys, y_inds, inv, counts = np.unique(y,return_index=True, return_inverse=True, return_counts=True)
	counts = counts.astype(np.uint32)
	# print(u_ys)
	# print(y_inds)
	# print(inv)
	# print(counts)
	# print(counts.dtype)


	entropy = entropy_func(counts)
	# print(entropy)

	split_tree(x,y_inds,entropy,entropy_func,counts)
	# binary_decision_tree_Woop()



# cdef binary_decision_tree_Woop():
# 	split_tree(gini)
# def 
	# return 

# def binary_decision_tree(bool[:,:] x, ):
	

	# print(gini(x))






