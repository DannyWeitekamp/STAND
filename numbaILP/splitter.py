from numbaILP.structref import define_structref, define_structref_template
from numbaILP.utils import _struct_from_pointer, _pointer_from_struct, _pointer_from_struct_incref, _decref_pointer, _decref_structref
from numba.experimental.structref import new
import numpy as np
import numba
from numba import types, njit, guvectorize,vectorize,prange, jit, literally
# from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType,ListType, unicode_type, NamedTuple,NamedUniTuple,Tuple,literal
from collections import namedtuple
import timeit
from sklearn import tree as SKTree
import os 
from operator import itemgetter

from numba import config, njit, threading_layer
# from numba.np.ufunc.parallel import _get_thread_id
from sklearn.preprocessing import OneHotEncoder
from numbaILP.fnvhash import hasharray#, AKD#, akd_insert,akd_get

from numbaILP.tree_structs import *
from numbaILP.data_stats import *
from numbaILP.split_caches import *
# from numbaILP.m_patch_array_hash import *
from numbaILP.akd import new_akd, AKDType


import logging
import time


# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.DEBUG)

config.THREADING_LAYER = 'thread_safe'
print("n threads", config.NUMBA_NUM_THREADS)
# os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '1'


CRITERION_none = 0
CRITERION_gini = 1
CRITERION_entropy = 2


######### Split Choosers ##########

# class SPLIT_CHOICE(IntEnum):
#   single_max = 1
#   all_max = 2

@njit(i8[::1](f8[::1]),nogil=True,fastmath=True,cache=True)
# @njit(i8[::1](f4[::1]),nogil=True,fastmath=True,cache=True,inline='never')
def choose_single_max(impurity_decrease):
    '''A split chooser that expands greedily by max impurity 
        (i.e. this is the chooser for typical decision trees)'''
    return np.asarray([np.argmax(impurity_decrease)])

@njit(i8[::1](f8[::1]),nogil=True,fastmath=True,cache=True)
# @njit(i8[::1](f4[::1]),nogil=True,fastmath=True,cache=True,inline='never')
def choose_all_max(impurity_decrease):
    '''A split chooser that expands every decision tree 
        (i.e. this chooser forces to build whole option tree)'''
    m = np.max(impurity_decrease)
    return np.where(impurity_decrease==m)[0]


split_choosers = {
    "single_max" : choose_single_max,
    "all_max"  : choose_all_max
}


# SPLIT_CHOICE_single_max = 1
# SPLIT_CHOICE_all_max = 2

# @njit(cache=True,inline='never')
# def split_chooser(func_enum,impurity_decrease):
#     if(func_enum == 1):
#         return choose_single_max(impurity_decrease)
#     elif(func_enum == 2):
#         return choose_all_max(impurity_decrease)
#     return choose_single_max(impurity_decrease)

PRED_CHOICE_majority = 1
PRED_CHOICE_pure_majority = 2
PRED_CHOICE_majority_general = 3
PRED_CHOICE_pure_majority_general = 4

N = 100
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

@njit(f8(u4,u4[:]), cache=True)#,inline='always')
def gini(total,counts):
    if(total > 0):
        s = 0.0
        for c_i in counts:
            prob = c_i / total;
            s += prob * prob 
        return 1.0 - s
    else:
        return 0.0


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


# counts_cache_fields = [
#     ('v_count_left', i8),
#     ('X', i4[:,:]),
#     ('Y', i4[:]),
# ]


@njit(cache=True)
def _fill_nominal_impurities(tree, splitter_context, split_cache, n_vals_j, k_j):
    b_ft_val = 0 
    v_counts       = split_cache.v_counts
    y_counts_per_v = split_cache.y_counts_per_v

    impurity = splitter_context.impurity
    impurities = splitter_context.impurities
    y_counts = splitter_context.y_counts
    n_samples = len(splitter_context.sample_inds)
    #If this feature is found to be constant then skip computing impurity
    if(np.sum(v_counts > 0) <= 1):
        # print("ZZAB")
        # split_cache.best_v = 0
        impurities[k_j,0] = impurity
        impurities[k_j,1] = impurity
        impurities[k_j,2] = impurity
    else:
        # print("ZZBB")
        b_imp_tot, b_imp_l, b_imp_r = np.inf, 0, 0,
        for ft_val in range(n_vals_j):
            counts_r = y_counts_per_v[ft_val]
            total_r = np.sum(counts_r)
            # print("Z",ft_val, y_counts, counts_r)

            counts_l = y_counts-counts_r
            total_l = n_samples-total_r

            # print("Z",total_l, counts_l)

            imp_l = gini(total_l, counts_l)
            imp_r = gini(total_r, counts_r)

            # print("Z1",ft_val)

            imp_tot = ((total_l/n_samples) * imp_l) + ((total_r/n_samples) * imp_r)
            if(imp_tot < b_imp_tot):
                b_imp_tot, b_imp_l, b_imp_r, b_ft_val = imp_tot, imp_l, imp_r, ft_val
        # print("ZZCB")            
        impurities[k_j,0] = b_imp_tot
        impurities[k_j,1] = b_imp_l
        impurities[k_j,2] = b_imp_r

    # print("ZZC")
    # split_cache.prev_best_v = split_cache.best_v
    split_cache.best_v = b_ft_val


@njit(cache=True,parallel=False)
def update_nominal_impurities(tree, splitter_context, iterative):
    #Make various variables local
    ds = tree.data_stats
    sc = splitter_context

    # n_samples, start, end = sc.n_samples, sc.start, sc.end
    X, Y = ds.X_nom, ds.Y
    # print("??", X.shape)
    n_vals = ds.n_vals
    # n_samples = ds.n_samples
    n_classes = ds.n_classes
    
    # counts_imps = sc.counts_imps
    
    # feature_inds = sc.feature_inds
    # n_const_fts = sc.n_const_fts
    sample_inds = sc.sample_inds
    n_samples = len(sc.sample_inds)


    # If in iterative mode then only update from where left off 
    if(iterative):
        # print("last_update", sc.n_last_update, n_samples)
        
        if(sc.n_last_update == n_samples): return
        # print("BEF",sample_inds)
        sample_inds = sample_inds[sc.n_last_update:]
        # last_update_ind = 0
        # for i in range(0, n_samples,-1):
        #     if(sample_inds[i] < sc.n_last_update):
        #         last_update_ind = i+1
        #         break
        # print(last_update_ind)
        # sample_inds = sample_inds[last_update_ind:]
        # print("AFT", sample_inds, sc.n_last_update)

    y_counts = sc.y_counts
    impurity = sc.impurity
    

    # Grow the count cache if the number of features has increased
    len_cache = len(sc.nominal_split_cache_ptrs)
    if(len_cache < X.shape[1]):
        new_sp_ptrs = np.zeros((X.shape[1],),dtype=np.int64)
        new_sp_ptrs[:len_cache] = sc.nominal_split_cache_ptrs
        sc.nominal_split_cache_ptrs = new_sp_ptrs

    # X_inds = X[inds]

    # y_count_left = counts_imps[:]

    # n_non_const = len(feature_inds)#-n_const_fts
    # print(len(feature_inds), n_const_fts, n_non_const)
    # print("ZA")
    sc.impurities = impurities = np.empty((X.shape[1],3),dtype=np.float64)
    # b_split, b_split_imp_total = 0, np.inf
    #Go through the samples in Fortran order (i.e. feature then sample)
    # for k_j in prange(0,n_non_const):
    for k_j in prange(X.shape[1]):
        # print(_get_thread_id(),k_j)
        j = k_j#feature_inds[k_j]
        n_vals_j = n_vals[j]
        cache_ptr = sc.nominal_split_cache_ptrs[j]
        # print(k_j, cache_ptr, n_vals_j, n_classes)
        if(cache_ptr != 0):

            split_cache = _struct_from_pointer(NominalSplitCacheType, cache_ptr)
            split_cache_shape = split_cache.y_counts_per_v.shape
            if(split_cache_shape[0] != n_vals_j or split_cache_shape[1] != n_classes):
                # print("EXPAND")
                expand_nominal_split_cache(split_cache, n_vals_j, n_classes)
            # print("reusing", k_j, split_cache.v_counts)
        else:
            split_cache = NominalSplitCache_ctor(n_vals_j, n_classes)
            sc.nominal_split_cache_ptrs[j] = _pointer_from_struct_incref(split_cache)

        # print(cache_ptr,sc.nominal_split_cache_ptrs[j])

        # print("ZB")
        v_counts       = split_cache.v_counts
        y_counts_per_v = split_cache.y_counts_per_v
        # print("BEF", y_counts_per_v, v_counts)
        # else:
        # y_counts_per_feature = np.zeros((n_vals_j,n_classes),dtype=np.uint32)
        # v_counts_per_feature = np.zeros((n_vals_j),dtype=np.uint32)


        #Update the feature counts for labels and values
        # for k_i in range(start, end):
        # print(Y,sample_inds)
        for i in sample_inds:
            # i = sample_inds[k_i]
            y_i = Y[i]
            y_counts_per_v[X[i,j],y_i] += 1
            v_counts[X[i,j]] += 1
            # for c in range(n_vals_j):
        # print("ZZB")
        # print(k_j, "::", v_counts, y_counts_per_v)
        _fill_nominal_impurities(tree, sc, split_cache, n_vals_j, j)


     # = impurities
    sc.n_last_update = n_samples
    # print(impurities)
    # print("ZC")
    # print(best_ind)

u4_arr = u4[:]
# empty_u8 = np.zeros(0,dtype=np.uint64)

@njit(cache=True)
def build_root(tree, iterative=False):
    ds = tree.data_stats
    Y = ds.Y
    sample_inds = np.arange(len(Y),dtype=np.uint32)

    impurity = gini(len(Y), ds.y_counts)
    
    #Make Root Node
    node_dict = new_akd(u4_arr,i4)
    nodes = List.empty_list(TreeNodeType)
    node = TreeNode_ctor(TTYPE_NODE,i4(0),ds.y_counts)
    nodes.append(node)
    tree.nodes = nodes

    empty_u8 = np.zeros(0,dtype=np.uint64)
    #Make Root Context
    if(iterative and empty_u8 in tree.context_cache):
        c = tree.context_cache[empty_u8]
    else:
        c = SplitterContext_ctor(empty_u8)
    
    reinit_splittercontext(c, node, sample_inds, ds.y_counts, impurity)    
    if(tree.ifit_enabled): tree.context_cache[empty_u8] = c

    context_stack = List.empty_list(SplitterContextType)
    context_stack.append(c)

    return context_stack, node_dict, nodes


# @njit(cache=True)
# def choose_next_splits():
#     pass


# ###### Array Keyed Dictionaries ######

# # BE = Tuple([u1[::1],i4])
# # BE_List = ListType(BE)

# @njit(nogil=True,fastmath=True)
# def akd_insert(akd,_arr,item,h=None):
#     '''Inserts an i4 item into the dictionary keyed by an array _arr'''
#     arr = _arr.view(np.uint8)
#     if(h is None): h = hasharray(arr)
#     elems = akd.get(h,List.empty_list(BE))
#     is_in = False
#     for elem in elems:
#         if(len(elem[0]) == len(arr) and
#             (elem[0] == arr).all()): 
#             is_in = True
#             break
#     if(not is_in):
#         elems.append((arr,item))
#         akd[h] = elems

# @njit(nogil=True,fastmath=True)
# def akd_get(akd,_arr,h=None):
#     '''Gets an i4 from a dictionary keyed by an array _arr'''
#     arr = _arr.view(np.uint8)
#     if(h is None): h = hasharray(arr) 
#     if(h in akd):
#         for elem in akd[h]:
#             if(len(elem[0]) == len(arr) and
#                 (elem[0] == arr).all()): 
#                 return elem[1]
#     return -1


TTYPE_NODE = u1(1)
TTYPE_LEAF = u1(2)

@njit(cache=True)
def next_split_chain(c, is_right, is_cont, split, val):
    ''' Make a new split chain by adding an encoding of 
        'is_right', 'is_cont', 'split', and 'val' to the 
        end of the previous split chain.
    '''
    l = len(c.split_chain)
    split_chain = np.empty(l+1,dtype=np.uint64)
    split_chain[:l] = c.split_chain
    split_chain[l] = u8((is_cont << 63) | (is_right << 62) | (split << 32) | val)
    return split_chain



@njit(cache=True)
def new_node(locs, tree, sample_inds, y_counts, impurity, is_right):
    ''' Creates a new node and a new context to compute its child nodes'''
    c, best_split, best_val,  iterative,  node_dict, context_stack, cache_nodes = locs
    nodes = tree.nodes
        # node_dict,nodes,new_contexts,cache_nodes = locs
        # NODE, LEAF = i4(1), i4(2) #np.array(1,dtype=np.int32).item(), np.array(2,dtype=np.int32).item()
    node_id = i4(-1)
    if (cache_nodes): node_id = node_dict.get(sample_inds,-1)
    # if (cache_nodes): node_id= akd_get(node_dict, sample_inds)
    if(node_id == -1):
        node_id = i4(len(nodes))
        # if(cache_nodes): akd_insert(node_dict, sample_inds, node_id)
        if(cache_nodes): node_dict[sample_inds] = node_id
        if(impurity > 0.0):
            node = TreeNode_ctor(TTYPE_NODE, node_id, y_counts)
            nodes.append(node)

            split_chain = next_split_chain(c, is_right, 0, best_split, best_val)
            # print('split_chain', split_chain, best_split)
            if(iterative and split_chain in tree.context_cache):
                new_c = tree.context_cache[split_chain]
                # print(new_c)
                ok = np.array_equal(new_c.split_chain, split_chain)
                # print("ALL OK", ok)
                # if(not ok):
                #     print(new_c.split_chain)
                #     print(split_chain)
                #     print(hash(new_c.split_chain), hash(split_chain))
                
            else:     
                new_c = SplitterContext_ctor(split_chain)
                if(tree.ifit_enabled): tree.context_cache[split_chain] = new_c

            reinit_splittercontext(new_c, node, sample_inds, y_counts, impurity)
            context_stack.append(new_c)
        else:
            nodes.append(TreeNode_ctor(TTYPE_LEAF, node_id, y_counts))
    return node_id







@njit(cache=True)
def extract_nominal_split_info(tree, c, split, iterative=False):
    ds = tree.data_stats
    bst_imps = c.impurities[split]
    imp_tot, imp_l, imp_r = bst_imps[0], bst_imps[1], bst_imps[2]
    # print("\nQ", split, c.nominal_split_cache_ptrs[split])
    # print(c.split_chain)

    splt_c = _struct_from_pointer(NominalSplitCacheType, c.nominal_split_cache_ptrs[split])

    best_v = splt_c.best_v
    # print(splt_c.y_counts_per_v)
    # print(splt_c.best_v)
    y_counts_r = splt_c.y_counts_per_v[splt_c.best_v]
    y_counts_l = c.y_counts - y_counts_r
    # print("P")
    # print(tree.data_stats.n_samples,c.y_counts,y_counts_l, y_counts_r)
    n_l = np.sum(y_counts_l)
    n_r = np.sum(y_counts_r)

    # print("POOP", splt_c.best_v, splt_c.prev_best_v)

    recalc_all = (splt_c.best_v != splt_c.prev_best_v)

    # Ensure inds_l and inds_r are large enough
    prev_n_l, prev_n_r = 0,0
    if(not tree.ifit_enabled):
        inds_l = np.empty(n_l, dtype=np.uint32)
        inds_r = np.empty(n_r, dtype=np.uint32)
    else:
        if(recalc_all):
            # print("INIT", splt_c.prev_best_v, "->", splt_c.best_v)
            splt_c.l_inds_buffer = np.empty(max(8,n_l*2),dtype=np.uint32)
            splt_c.r_inds_buffer = np.empty(max(8,n_r*2),dtype=np.uint32)
        else:
            # print("UPD", splt_c.best_v)
            prev_n_l = len(splt_c.l_inds)
            prev_n_r = len(splt_c.r_inds)

            if(n_l > len(splt_c.l_inds_buffer)):
                buff = np.empty(n_l*2,dtype=np.uint32)
                buff[:prev_n_l] = splt_c.l_inds
                splt_c.l_inds_buffer = buff

            if(n_r > len(splt_c.r_inds_buffer)):
                buff = np.empty(n_r*2,dtype=np.uint32)
                buff[:prev_n_r] = splt_c.r_inds
                splt_c.r_inds_buffer = buff

        inds_l = splt_c.l_inds = splt_c.l_inds_buffer[:n_l]
        inds_r = splt_c.r_inds = splt_c.r_inds_buffer[:n_r]

    # print("GOOP")

    # Find sample_inds, the set of instance inds we need to update counts for. 
    if(not iterative):
        sample_inds = c.sample_inds
        p_l, p_r = 0, 0
    else:
        # print(splt_c.n_last_update, len(c.sample_inds))
        sample_inds = c.sample_inds if(recalc_all) \
                        else c.sample_inds[splt_c.n_last_update:]
        p_l, p_r = prev_n_l, prev_n_r

    # print("BE", p_l,":", n_l,",", p_r,":", n_r, len(splt_c.l_inds_buffer), len(splt_c.r_inds_buffer))
    # print("UPDATED", splt_c.n_last_update, len(sample_inds), len(c.sample_inds))
    # print("XXX", n_l, n_r, inds_l, inds_r)
    # print(splt_c.best_v, splt_c.prev_best_v)

    # Append to inds_l and inds_r
    # print(ds.X_nom)
    # print("<<", split, '==', splt_c.best_v, sample_inds, ds.X_nom[sample_inds, split])
    for ind in sample_inds:
        # print(ind)
        # print(ds.X_nom[ind, split])
        if (ds.X_nom[ind, split]==splt_c.best_v):
            inds_r[p_r] = ind
            p_r += 1
        else:
            inds_l[p_l] = ind
            p_l += 1

    # print("AF", p_l ,":", n_l, ",", p_r,":", n_r)

    # print(c.sample_inds)
    # print(sample_inds)
    
    # NOTE: Can problably delete     
    if(p_l != n_l or p_r != n_r):
        raise RuntimeError("Failed to fully update counts.")
    splt_c.n_last_update = len(c.sample_inds)
    splt_c.prev_best_v = splt_c.best_v

    
    
    # print("P")
    # print(inds_l, inds_r)

    return inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, best_v
            



@njit(cache=True, locals={'y_counts_l' : u4[:], 'y_counts_r' : u4[:]})
def fit_tree(tree, config, iterative=False):
    '''
    Refits the tree from its DataStats
    '''
    cache_nodes = False
    # print("Z")
    context_stack, node_dict, nodes =  \
        build_root(tree)
    
    # print("A")
    

    while(len(context_stack) > 0):
        # print("AZ")
        c = context_stack.pop()
        update_nominal_impurities(tree, c ,iterative)
        # print("BZ")
        # print(c.impurities[:,0],c.start,c.end)

        best_splits = tree.split_chooser(c.impurity-c.impurities[:,0])

        # best_split = np.argmin(c.impurity-c.impurities[:,0])
        # print("---")
        for split in best_splits:
            # print("S", split)

            inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, val = \
                extract_nominal_split_info(tree, c, split, iterative)

            # print("S1", split, inds_l, inds_r, val, "\n")

            # if(impurity_decrease[split] <= 0.0):
            

            if(c.impurity - imp_tot <= 0):
                c.node.ttype = TTYPE_LEAF

            else:
                # print("S2", split)
                ptr = _pointer_from_struct(c)
                locs = (c, split, val, iterative, node_dict, context_stack, cache_nodes)
                node_l = new_node(locs, tree, inds_l, y_counts_l, imp_l, 0)
                node_r = new_node(locs, tree, inds_r, y_counts_r, imp_r, 1)

                split_data = SplitData(u1(False),i4(split), i4(val), i4(node_l), i4(node_r))
                #np.array([split, val, node_l, node_r, -1],dtype=np.int32)
                c.node.split_data.append(split_data)
                c.node.op_enum = OP_EQ

        # print("B")
        if(not iterative):
            SplitterContext_dtor(c)
        # print("C")
        
        # _decref_pointer(ptr)
    return 0
    # print("DONE")
    # return Tree(nodes,data_stats.u_ys)

            

######### Prediction Choice Functions #########
# class PRED_CHOICE(IntEnum):
#   majority = 1
#   pure_majority = 2
#   majority_general = 3
#   pure_majority_general = 4



@njit(nogil=True,fastmath=True,cache=True,inline='never')
def get_pure_counts(leaf_counts):
    pure_counts = List()
    for count in leaf_counts:
        if(np.count_nonzero(count) == 1):
            pure_counts.append(count)
    return pure_counts

@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_majority(leaf_counts,positive_class):
    ''' If multiple leaves on predict (i.e. option tree), choose 
        the class predicted by the majority of leaves.''' 
    predictions = np.empty((len(leaf_counts),),dtype=np.int32)
    for i,count in enumerate(leaf_counts):
        predictions[i] = np.argmax(count)
    c,u, inds = unique_counts(predictions)
    _i = np.argmax(c)
    return u[_i]

@njit(nogil=True,fastmath=True,cache=True,inline='never')
def choose_pure_majority(leaf_counts,positive_class):
    ''' If multiple leaves on predict (i.e. option tree), choose 
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

@njit(nogil=True, fastmath=True,cache=True, inline='never')
def pred_choice_func(leaf_counts, cfg):
    if(cfg.pred_choice_enum == 1):
        return choose_majority(leaf_counts, cfg.positive_class)
    elif(cfg.pred_choice_enum == 2):
        return choose_pure_majority(leaf_counts, cfg.positive_class)
    elif(cfg.pred_choice_enum == 3):
        return choose_majority_general(leaf_counts, cfg.positive_class)
    elif(cfg.pred_choice_enum == 4):
        return choose_pure_majority_general(leaf_counts, cfg.positive_class)
    return choose_majority(leaf_counts, cfg.positive_class)

@njit(nogil=True,fastmath=True, cache=True, locals={"ZERO":u1, "TO_VISIT":u1, "VISITED": u1, "_n":i4})
def predict_tree(tree, x_nom, x_cont, config):#, pred_choice_enum, positive_class=0,decode_classes=True):
    '''Predicts the class associated with an unlabelled sample using a fitted 
        decision/option tree'''
    ZERO, TO_VISIT, VISITED = 0, 1, 2
    L = max(len(x_nom),len(x_cont))
    out = np.empty((L,),dtype=np.int64)
    y_uvs = tree.data_stats.u_ys#_get_y_order(tree)
    nom_v_maps = tree.data_stats.nom_v_maps
    for i in range(L):
        # Use a mask instead of a list to avoid repeats that can blow up
        #  if multiple splits are possible. Keep track of visited in case
        #  of loops (Although there should not be any loops).
        new_node_mask = np.zeros((len(tree.nodes),),dtype=np.uint8)
        new_node_mask[0] = TO_VISIT
        nodes_to_visit = np.nonzero(new_node_mask==TO_VISIT)[0]
        # print(node_inds)
        leafs = List()

        while len(nodes_to_visit) > 0:
            # Go through every node that has been queued for a visit. In a traditional
            #  decision tree there should only ever be one next node.
            # print(nodes_to_visit)
            for ind in nodes_to_visit:
                node = tree.nodes[ind]
                op = node.op_enum
                if(node.ttype == TTYPE_NODE):
                    # Test every split in the node. Again in a traditional decision tree
                    #  there should only be one split per node.
                    for sd in node.split_data:
                        # split_on, ithresh, left, right, nan  = s[0],s[1],s[2],s[3],s[4]

                        # Determine if this sample should feed right, left, or nan (if ternary)
                        if(not sd.is_continous):
                            # Nominal case
                            mapped_val = nom_v_maps[sd.split_ind][x_nom[i,sd.split_ind]]
                            if(mapped_val==sd.val):
                                _n = sd.right
                            else:
                                _n = sd.left
                        else:
                            #Need to reimplement
                            pass
                        # else:
                        #     # Continous case
                        #     thresh = np.int32(ithresh).view(np.float32)
                        #     j = split_on-xb.shape[1] 

                        #     if(exec_op(op,x_cont[i,j],thresh)):
                        #         _n = right
                        #     else:
                        #         _n = left
                        if(new_node_mask[_n] != VISITED): new_node_mask[_n] = TO_VISIT
                            
                else:
                    leafs.append(node.counts)

            #Mark all nodes_to_visit as visited so we don't mark them for a revisit
            for ind in nodes_to_visit:
                new_node_mask[ind] = VISITED

            nodes_to_visit = np.nonzero(new_node_mask==TO_VISIT)[0]
        # print(leafs)
        # Since the leaf that the sample ends up in is ambiguous for an ambiguity
        #   tree we need a subroutine that chooses how to classify the sample from the
        #   various leaves that it could end up in. 
        # print(i, leafs)
        out_i = pred_choice_func(leafs, config)
        # if(decode_classes):out_i = y_uvs[out_i]
        out_i = y_uvs[out_i]
        out[i] = out_i
    return out





def str_op(op_enum):
    if(op_enum == OP_EQ):
        return "=="
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
    l = ["TREE w/ classes: %s"%tree.data_stats.u_ys]
    nom_v_inv_maps = tree.data_stats.nom_v_inv_maps
    # node_offset = 1
    # while node_offset < tree[0]:
    for node in tree.nodes:
        # node_width = tree[node_offset]
        ttype, index, splits, counts = node.ttype, node.index, node.split_data, node.counts#_unpack_node(tree,node_offset)
        op = node.op_enum
        if(ttype == TTYPE_NODE):
            s  = "NODE(%s) : " % (index)
            for sd in splits:
                if(not sd.is_continous): #<-A threshold of 1 means it's binary
                    inv_map = nom_v_inv_maps[sd.split_ind]
                    s += f"({sd.split_ind},=={inv_map[sd.val]})[L:{sd.left} R:{sd.right}"
                else:
                    thresh = np.int32(sd.val).view(np.float32) if op != OP_EQ else np.int32(sd.val)
                    instr = str_op(op)+str(thresh) if op != OP_ISNAN else str_op(op)
                    s += f"({sd.split_ind},{instr})[L:{sd.left} R:{sd.right}"
                    # s += "(%s,%s)[L:%s R:%s" % (sd.split_ind,instr,sd.left,sd.right)
                s += "] "# if(split[4] == -1) else ("NaN:" + str(split[4]) + "] ")
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
    'option_tree' : {
        'criterion' : 'gini',
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

config_fields = [
    ('criterion_enum', types.Any),
    # ('total_enum', types.Any),
    ('split_choice_enum', types.Any),
    ('pred_choice_enum', types.Any),
    ('cache_nodes', types.Any),
    ('sep_nan', types.Any),
    ('positive_class', i4),
]

TreeClassifierConfig, TreeClassifierConfigTemplate = define_structref_template("TreeClassifierConfig", config_fields, define_constructor=False)

@njit(cache=True)
def new_config(typ):
    return new(typ)

class TreeClassifier(object):
    def __init__(self, preset_type='decision_tree', 
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
        # total_enum = g.get(f"TOTAL_{total_func}",None)
        # split_choice_enum = g.get(f"SPLIT_CHOICE_{split_choice}",None)
        pred_choice_enum = g.get(f"PRED_CHOICE_{pred_choice}",None)

        if(criterion_enum is None): raise ValueError(f"Invalid criterion {criterion}")
        # if(total_enum is None): raise ValueError(f"Invalid criterion {total_func}")
        # if(split_choice_enum is None): raise ValueError(f"Invalid split_choice {split_choice}")
        if(pred_choice_enum is None): raise ValueError(f"Invalid pred_choice {pred_choice}")
        self.positive_class = positive_class

        config_dict = {k:v for k,v in config_fields}
        config_dict['criterion_enum'] = literal(criterion_enum)
        config_dict['split_choice_enum'] = literal(1)
        config_dict['pred_choice_enum'] = literal(pred_choice_enum)

        ConfigType = TreeClassifierConfigTemplate([(k,v) for k,v in config_dict.items()])

        # print(config_dict)

        # self.config = 

        tf_dict = {k:v for k,v in tree_fields}
        tf = [(k,v) for k,v in {**tf_dict, **{"ifit_enabled": literal(True)}}.items()]
        self.tree_type = TreeTypeTemplate(tf)
        self.config = new_config(ConfigType)

        self.tree = Tree_ctor(self.tree_type, split_choosers[split_choice])

        # print(self.tree)

        # @njit(cache=True)
        # def _fit(xb,xc,y,miss_mask,ft_weights): 
        #     out =fit_tree(xb,xc,y,miss_mask,
        #             ft_weights=ft_weights,
        #             # missing_values=missing_values,
        #             criterion_enum=literally(criterion_enum),
        #             total_enum=literally(total_enum),
        #             split_enum=literally(split_enum),
        #             positive_class=positive_class,
        #             criterion_enum2=literally(criterion_enum2),
        #             total_enum2=literally(total_enum2),
        #             sep_nan=literally(sep_nan),
        #             cache_nodes=literally(cache_nodes)
        #          )
        #     return out
        # self._fit = _fit

        # @njit(cache=True)
        # def _inf_gain(xb,xc,y,miss_mask,ft_weights):    
        #     out =inf_gain(xb,xc,y,miss_mask,
        #             ft_weights=ft_weights,
        #             # missing_values=missing_values,
        #             criterion_enum=literally(criterion_enum),
        #             total_enum=literally(total_enum),
        #             positive_class=positive_class,
        #          )
        #     return out
        # self._inf_gain = _inf_gain

        # @njit(cache=True)
        # def _predict(tree, xb, xc, positive_class): 
        #     out =predict_tree(tree,xb,xc,
        #             pred_choice_enum=literally(pred_choice_enum),
        #             positive_class=positive_class,
        #             decode_classes=True
        #          )
        #     return out
        # self._predict = _predict
        
        
    def fit(self, X_nom, X_cont, Y, miss_mask=None, ft_weights=None):
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        # if(miss_mask is None): miss_mask = np.zeros_like(xc, dtype=np.bool)
        # if(ft_weights is None): ft_weights = np.empty(xb.shape[1]+xc.shape[1], dtype=np.float64)
        if(X_nom.ndim != 2): raise ValueError(f"X_nom shoud be 2 dimensional, got shape {X_nom.shape}")
        if(X_cont.ndim != 2): raise ValueError(f"X_cont shoud be 2 dimensional, got shape {X_cont.shape}")
        if(Y.ndim != 1): raise ValueError(f"Y shoud be 1 dimensional, got shape {Y.shape}")

        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        Y = Y.astype(np.int32)
        # miss_mask = miss_mask.astype(np.bool)
        # ft_weights = ft_weights.astype(np.float64)
        # assert miss_mask.shape == xc.shape

        # self.tree = self._fit(xb, xc, y, miss_mask, ft_weights)
        # self.tree.data_stats = DataStats_ctor()
        # clear_tree_datastats(self.tree)
        # print("A")
        # print(X_nom,X_nom.dtype)
        reinit_tree_datastats(self.tree, X_nom, X_cont, Y)
        # print("B")
        fit_tree(self.tree, self.config, False)
        # print("C")

    # def inf_gain(self,xb,xc,y,miss_mask=None, ft_weights=None):
    #     if(xb is None): xb = np.empty((0,0), dtype=np.uint8)
    #     if(xc is None): xc = np.empty((0,0), dtype=np.float64)
    #     if(miss_mask is None): miss_mask = np.zeros_like(xc, dtype=np.bool)
    #     if(ft_weights is None): ft_weights = np.empty(xb.shape[1]+xc.shape[1], dtype=np.float64)
    #     xb = xb.astype(np.uint8)
    #     xc = xc.astype(np.float64)
    #     y = y.astype(np.int64)
    #     miss_mask = miss_mask.astype(np.bool)
    #     ft_weights = ft_weights.astype(np.float64)
    #     # assert miss_mask.shape == xc.shape
    #     return self._inf_gain(xb, xc, y, miss_mask, ft_weights)

    def ifit(self, x_nom, x_cont, y, miss_mask=None, ft_weights=None):
        if(x_nom is None): x_nom = np.empty((0,), dtype=np.int32)
        if(x_cont is None): x_cont = np.empty((0,), dtype=np.float32)

        # self.tree.data_stats = DataStats_ctor()
        update_data_stats(self.tree.data_stats, x_nom, x_cont, y)
        fit_tree(self.tree, self.config, True)

        


    def predict(self, X_nom, X_cont, positive_class=None):
        if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
        if(positive_class is None): positive_class = self.positive_class
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        return predict_tree(self.tree,X_nom, X_cont,self.config)
        # return self._predict(self.tree, xb, xc, positive_class)

    def __str__(self):
        return str_tree(self.tree)

    def as_conditions(self,positive_class=None, only_pure_leaves=False):
        if(positive_class is None): positive_class = self.positive_class
        return tree_to_conditions(self.tree, positive_class, only_pure_leaves)

# TreeClassifier()

@njit(cache=True)
def build_XY(N=1000,M=100):
    p0 = np.array([1,1,1,0,0],dtype=np.int32)
    p1 = np.array([0,1,1,1,0],dtype=np.int32)
    p2 = np.array([0,0,1,1,1],dtype=np.int32)
    p3 = np.array([1,0,1,1,0],dtype=np.int32)
    p4 = np.array([1,0,1,0,1],dtype=np.int32)
    X = np.random.randint(0,5,(N,M)).astype(np.int32)
    Y = np.random.randint(0,3,(N,)).astype(np.int32)

    for x, y in zip(X,Y):
        if(y == 0): 
            x[:5] = np.where(p0,p0,x[:5])
        elif(y==1):
            x[:5] = np.where(p1,p1,x[:5])
        elif(y==2):
            x[:5] = np.where(p2,p2,x[:5])
        elif(y==3):
            x[:5] = np.where(p3,p3,x[:5])
        elif(y==4):
            x[:5] = np.where(p4,p4,x[:5])
    return X, Y


# X, Y = build_XY(10,10)
X, Y = build_XY()
X_cont = np.zeros((0,0),dtype=np.float32)
one_h_encoder = OneHotEncoder()
X_oh = one_h_encoder.fit_transform(X).toarray()
# @njit(cache=True)


def test_fit_tree():
    dt = TreeClassifier(preset_type='option_tree')
    dt.fit(X, X_cont, Y)
    # fit_tree(X, X_cont, Y)
    

def test_sklearn():
    clf = SKTree.DecisionTreeClassifier()
    clf.fit(X_oh,Y)



print(time_ms(test_fit_tree))
print(time_ms(test_sklearn))







dt = TreeClassifier(preset_type='option_tree')
# dt.fit(X, X_cont, Y)
# print(dt)
# print("printed")
# print_tree(tree)


# print("PRINT")

#### test_basics ####

def setup1():
    data1 = np.asarray([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3 0
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1 1
    [0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1 2
    [0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1 3
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2 4
    [0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2 5
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2 6
    ],np.int32);

    labels1 = np.asarray([3,1,1,1,2,2,2],np.int32);
    return data1, labels1

def setup2():
    data2 = np.asarray([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0,0,0,0,0], #1
    [0,0,1,0,0,0], #1
    [0,1,1,0,0,0], #1
    [1,1,1,0,0,1], #2
    [0,1,1,1,1,0], #2
    [1,1,1,0,1,0], #2
    ],np.int32);

    labels2 = np.asarray([1,1,1,2,2,2],np.int32);
    data2 = data2[:,[1,0,2,3,4,5]]
    return data2, labels2

def setup3():
    data3 = np.asarray([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0], #1
    [1,0], #1
    [0,1], #1
    [1,1], #2
    ],np.int32);

    labels3 = np.asarray([1,1,1,2],np.int32);
    return data3, labels3

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
print("PRINT")

X,Y = setup1()
dt.fit(X, X_cont, Y)
print(dt)
print("Pred:", dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))
# raise ValueError()
X,Y = setup2()
dt.fit(X, X_cont, Y)
# print("SQUIB")
print(dt)
print("Pred:", dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))



X,Y = setup3()
dt.fit(X, X_cont, Y)
print(dt)
print(dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))

print("------------------------------------")
#KEEP 
X,Y = build_XY(N=1000,M=100)
dt = TreeClassifier()
x_c = np.zeros((0,),dtype=np.float32)

t0 = time.time_ns()
for i,(x_n, y) in enumerate(zip(X, Y)):
    # print(i)
    # t = time.time_ns()
    dt.ifit(x_n, x_c, y)
    
print(f"avg {((time.time_ns()-t0)/1e6) / len(X)} ms")



# print("iFIT DONE")
# print(dt)

# dt_f = TreeClassifier()
# dt_f.fit(X,X_cont,Y)
# print(dt_f)

# print(dt.predict(X,X_cont))
# print(dt_f.predict(X,X_cont))

# print(dt_f.predict(X,X_cont) == dt.predict(X,X_cont))




class DecisionTree2(object):
    # def __init__(self, impl="decision_tree", use_missing=False):
    def __init__(self, impl="decision_tree_w_greedy_backup", use_missing=False):
        print("IMPL:",impl)
        if(impl == "sklearn"):
            self.x_format = "one_hot"
            self.is_onehot = True
            self.dt = DecisionTreeClassifier()
        else:
            self.x_format = "integer"
            self.is_onehot = False
            self.dt = TreeClassifier(impl)


        
        self.impl = impl
        self.X = []
        self.y = []
        self.slots = {}
        self.inverse = []
        self.slots_count = 0
        self.X_list = []
        self.use_missing = use_missing
        

    def _designate_new_slots(self,x):
        ''' Makes new slots for unseen keys and values'''
        for k, v in x.items():
            if(k not in self.slots):
                slot = self.slots_count if self.is_onehot else 0
                vocab = self.slots[k] = {chr(0) : slot}         
                self.slots_count += 1
                self.inverse.append(f'!{k}')
            else:
                vocab = self.slots[k]

            if(v not in vocab): 
                slot = self.slots_count if self.is_onehot else len(vocab)
                vocab[v] = slot
                self.slots_count += 1
                self.inverse.append(f'{k}=={v}')

    def _dict_to_onehot(self,x,silent_fail=False):
        x_new = [0]*self.slots_count
        for k, vocab in self.slots.items():
            # print(k, vocab)
            val = x.get(k,chr(0))
            if(silent_fail):
                if(val in vocab): x_new[vocab[val]] = 1
            else:
                x_new[vocab[val]] = 1
        return np.asarray(x_new,dtype=np.bool)

    def _dict_to_integer(self,x,silent_fail=False):
        x_new = [0]*len(self.slots)
        for i,(k, vocab) in enumerate(self.slots.items()):
            # print(k, vocab)
            val = x.get(k,chr(0))
            if(silent_fail):
                if(val in vocab): x_new[i] = vocab[val]
            else:
                x_new[i] = vocab[val]
        return np.asarray(x_new,dtype=np.int32)

    def _transform_dict(self,x,silent_fail=False):
        if(self.is_onehot):
            return self._dict_to_onehot(x,silent_fail)
        else:
            return self._dict_to_integer(x,silent_fail)

    # def _gen_feature_weights(self, strength=1.0):
    #     weights = [0]*(self.slots_count if self.x_format == "one_hot" else len(self.slots))
    #     for k, vocab in self.slots.items():
    #         # print(k, vocab)
    #         val = (1.0/max(len(vocab)-1,1)) if self.x_format == "one_hot" else 1.0
    #         w = (1.0-strength) + (strength * val)
    #         for val,ind in vocab.items():
    #             weights[ind] = w

    #     return np.asarray(weights,dtype=np.float64)


    def _compose_one_hots(self):
        X = np.empty( (len(self.X_list), self.slots_count), dtype=np.uint8)
        missing_vals = [None]*len(self.X_list)
        for i, one_hot_x in enumerate(self.X_list):
            X[i,:len(one_hot_x)] = one_hot_x
            X[i,len(one_hot_x):] = 2 if self.use_missing else 0 # missing
        return X

    def _compose_integers(self):
        X = np.empty( (len(self.X_list), len(self.slots)), dtype=np.int32)
        # missing_vals = [None]*len(self.X_list)
        for i, int_x in enumerate(self.X_list):
            X[i,:len(int_x)] = int_x
            X[i,len(int_x):] = 0
        return X


    

    def _compose(self):
        if(self.x_format == 'one_hot'):
            return self._compose_one_hots()
        else:
            return self._compose_integers()





    def ifit(self, x, y):
        self._designate_new_slots(x)        
        trans_x = self._transform_dict(x)


        self.X_list.append(trans_x)
        self.X = self._compose()

        
        self.y.append(int(y) if not isinstance(y, tuple) else y)
        Y = np.asarray(self.y,dtype=np.int64)

        self.fit(self.X,Y)

    def fit(self, X, Y):
        if(not isinstance(X, np.ndarray)):
            self.X_list = []
            for x in X:
                self._designate_new_slots(x)
                self.X_list.append(self._transform_dict(x))
            self.X = X = self._compose()

        Y = np.asarray(Y,dtype=np.int64)
        # print(X)
        if(self.impl == "sklearn"):
            return self.dt.fit(X, Y)
        else:
            # tree_str = str(self.dt) if getattr(self.dt, "tree",None) is not None else ''
            # [n.split_on for n in self.dt.tree.nodes]
            # inds = [int(x.split(" : (")[1].split(")")[0]) for x in re.findall(r'NODE.+',tree_str)]

            # print()
            # print("---", self.rhs, "---")
            # tree_condition_inds(self.dt.tree)
            # print(tree_str)

            
            # if(False):
            #     # ft_weights = self._gen_feature_weights()
            #     print(json.dumps({ind: str(self.inverse[ind])+f"(w:{ft_weights[ind]:.2f})" for ind in inds},indent=2)[2:-2])
            # else:
            #     ft_weights = np.ones((X.shape[1]),dtype=np.float64)
            #     print(json.dumps({ind: str(self.inverse[ind]) for ind in inds},indent=2)[2:-2])
            # print(json.dumps({ind: str(self.inverse[ind]) for ind in inds},indent=2)[2:-2])
            

            # return self.dt.fit(X, None, Y, None)
            X_cont = np.zeros((0,0),dtype=np.float32)
            return self.dt.fit(X, X_cont, Y)


    def predict(self, X):
        is_onehot = self.x_format == 'one_hot'
        width = self.slots_count if is_onehot else len(self.slots)
        dtype = np.bool if is_onehot else np.int32
        encoded_X = np.empty((len(X), width),dtype=dtype)

        for i, x in enumerate(X):
            encoded_x = self._transform_dict(x,silent_fail=True)
            encoded_X[i] = encoded_x

        if(self.impl == "sklearn"):
            pred = self.dt.predict(encoded_X)
        else:
            # print("PRED:",self.rhs, self.dt.predict(onehot_X,None))
            pred = self.dt.predict(encoded_X,None)

        return pred


# dt2 = DecisionTree2()

# dt2.ifit({'A': 0, 'B' : 0},0)
# dt2.ifit({'A': 0, 'B' : 1},0)
# dt2.ifit({'A': 1, 'B' : 0},0)
# dt2.ifit({'A': 1, 'B' : 1},1)


# print(dt2.predict([{'A': 0, 'B' : 0}]))
# print(dt2.predict([{'A': 0, 'B' : 1}]))
# print(dt2.predict([{'A': 1, 'B' : 0}]))
# print(dt2.predict([{'A': 1, 'B' : 1}]))

# print(dt2.dt)


#Notes
#For nominal values and class labels integers can be discontinuous
# we should ensure that they are, but requires using a dictionary
# or sorting the data 
