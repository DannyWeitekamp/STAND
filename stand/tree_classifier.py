from stand.structref import define_structref, define_structref_template
from stand.utils import _struct_from_pointer, _pointer_from_struct, _pointer_from_struct_incref, _decref_pointer, _decref_structref
from numba.experimental.structref import new
import numpy as np
import numba
from numba import types, njit, guvectorize,vectorize,prange, jit, literally
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType,ListType, unicode_type, NamedTuple,NamedUniTuple,Tuple,literal
from collections import namedtuple
import timeit
import os 
from operator import itemgetter
from numba import config, njit, threading_layer
from sklearn.preprocessing import OneHotEncoder
from stand.fnvhash import hasharray
from stand.tree_structs import *
from stand.data_stats import *
from stand.split_caches import *
from stand.akd import new_akd, AKDType

from numba.experimental.function_type import _get_wrapper_address


import logging
import time


config.THREADING_LAYER = 'thread_safe'


# --------------------------------
#  Impurity Functions

impurity_func_sig = f8(u4,u4[:])

@njit(impurity_func_sig, cache=True)
def gini_impurity(total, counts):
    if(total > 0):
        s = 0.0
        for c_i in counts:
            prob = c_i / total;
            s += prob * prob 
        return 1.0 - s
    else:
        return 0.0


impurity_funcs = {
    "gini" : gini_impurity,
    "entropy"  : None
}


# --------------------------------
#  Split Choosers

split_chooser_sig = i8[::1](f8[::1])

@njit(split_chooser_sig,nogil=True,fastmath=True,cache=True)
# @njit(i8[::1](f4[::1]),nogil=True,fastmath=True,cache=True,inline='never')
def choose_single_max(impurity_decrease):
    '''A split chooser that expands greedily by max impurity 
        (i.e. this is the chooser for typical decision trees)'''
    return np.asarray([np.argmax(impurity_decrease)])

@njit(split_chooser_sig,nogil=True,fastmath=True,cache=True)
# @njit(i8[::1](f4[::1]),nogil=True,fastmath=True,cache=True,inline='never')
def choose_all_max(impurity_decrease):
    '''A split chooser that expands every decision tree 
        (i.e. this chooser forces to build whole option tree)'''
    m = np.max(impurity_decrease)
    return np.where(impurity_decrease==m)[0]

@njit(split_chooser_sig,nogil=True,fastmath=True,cache=True)
# @njit(i8[::1](f4[::1]),nogil=True,fastmath=True,cache=True,inline='never')
def choose_all_near_max(impurity_decrease):
    '''A split chooser that expands every decision tree 
        (i.e. this chooser forces to build whole option tree)'''
    m = np.max(impurity_decrease)*.9
    return np.where(impurity_decrease>=m)[0]


split_choosers = {
    "single_max" : choose_single_max,
    "all_max"  : choose_all_max,
    "all_near_max"  : choose_all_near_max
}


# -----------------------------------------------------------------------------
# fit() and ifit()

@njit(cache=True)
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



@njit(cache=True)
def _fill_nominal_impurities(tree, splitter_context, split_cache, n_vals_j, k_j):
    b_ft_val = 0 
    v_counts       = split_cache.v_counts
    y_counts_per_v = split_cache.y_counts_per_v

    impurity = splitter_context.impurity
    impurities = splitter_context.impurities
    y_counts = splitter_context.y_counts
    n_samples = len(splitter_context.sample_inds)
    impurity_func = tree.impurity_func
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

            imp_l = impurity_func(u4(total_l), counts_l)
            imp_r = impurity_func(u4(total_r), counts_r)

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
    for j in prange(X.shape[1]):
        # print(_get_thread_id(),k_j)
        # j = k_j#feature_inds[k_j]
        n_vals_j = n_vals[j]
        cache_ptr = sc.nominal_split_cache_ptrs[j]
        # print(j, cache_ptr, n_vals_j, n_classes)
        if(cache_ptr != 0):

            split_cache = _struct_from_pointer(NominalSplitCacheType, cache_ptr)
            split_cache_shape = split_cache.y_counts_per_v.shape
            if(split_cache_shape[0] != n_vals_j or split_cache_shape[1] != n_classes):
                # print("EXPAND")
                expand_nominal_split_cache(split_cache, n_vals_j, n_classes)
            # print("reusing", j, split_cache.v_counts)
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

    impurity = tree.impurity_func(u4(len(Y)), ds.y_counts)
    
    #Make Root Node
    node = TreeNode_ctor(TTYPE_NODE,i4(0),sample_inds,ds.y_counts)

    # Make Sure various node containers are reset
    node_dict = new_akd(u4_arr,i4)    
    tree.nodes = List.empty_list(TreeNodeType)
    tree.nodes.append(node)
    tree.leaves = List.empty_list(TreeNodeType)

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

    return context_stack, node_dict


TTYPE_NODE = u1(1)
TTYPE_LEAF = u1(2)

@njit(cache=True)
def encode_split(is_cont, negated, split, val):
    return u8((is_cont << 63) | (negated << 62) | (split << 32) | val)


@njit(Tuple((u1,u1,i4,i4))(u8),cache=True)
def decode_split(enc_split):
    is_cont = enc_split >> 63
    negated = enc_split >> 62  & u8(1)
    split =  ((enc_split << 2) >> 34) & 0xFFFFFFFF
    val =     enc_split               & 0xFFFFFFFF
    return is_cont, negated, split, val

@njit(cache=True)
def extend_split_chain(c, encoded_split):
    ''' Make a new split chain by adding an encoding of 
        'is_right', 'is_cont', 'split', and 'val' to the 
        end of the previous split chain.
    '''
    l = len(c.split_chain)
    split_chain = np.empty(l+1,dtype=np.uint64)
    split_chain[:l] = c.split_chain
    split_chain[l] = encoded_split
    return split_chain



@njit(cache=True)
def new_node(locs, tree, sample_inds, y_counts, impurity, is_right):
    ''' Creates a new node and a new context to compute its child nodes'''
    c, best_split, best_val,  iterative,  node_dict, context_stack = locs
    nodes = tree.nodes
    node_id = i4(-1)
    if (tree.cache_nodes): 
        node_id = node_dict.get(sample_inds,-1)
    # print(node_id, "<<", sample_inds)

    encoded_split = encode_split(0, not is_right, best_split, best_val)

    if(node_id == -1):
        node_id = i4(len(nodes))        
        # if(cache_nodes): akd_insert(node_dict, sample_inds, node_id)
        if(tree.cache_nodes): node_dict[sample_inds] = node_id
        if(impurity > 0.0):
            node = TreeNode_ctor(TTYPE_NODE, node_id, sample_inds, y_counts)

            split_chain = extend_split_chain(c, encoded_split)
            if(iterative and split_chain in tree.context_cache):
                new_c = tree.context_cache[split_chain]                
            else:     
                new_c = SplitterContext_ctor(split_chain)
                if(tree.ifit_enabled): tree.context_cache[split_chain] = new_c

            reinit_splittercontext(new_c, node, sample_inds, y_counts, impurity)
            context_stack.append(new_c)
        else:
            node = TreeNode_ctor(TTYPE_LEAF, node_id, sample_inds, y_counts)
            tree.leaves.append(node)

        tree.nodes.append(node)
    else:
        node = tree.nodes[node_id]

    node.parents.append((c.node.index, encoded_split))
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
    # if(p_l != n_l or p_r != n_r):
    #     raise RuntimeError("Failed to fully update counts.")
    splt_c.n_last_update = len(c.sample_inds)
    splt_c.prev_best_v = splt_c.best_v

    
    
    # print("P")
    # print(inds_l, inds_r)

    return inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, best_v
            



@njit(cache=True, locals={'y_counts_l' : u4[:], 'y_counts_r' : u4[:]})
def fit_tree(tree, iterative=False):
    '''
    Refits the tree from its DataStats
    '''
    context_stack, node_dict =  \
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
            
            # This prevents nodes already known to be leaves from being added
            #  to the set of leaves. Not sure why cannot check this outside of loop. 
            if(c.node.ttype == TTYPE_LEAF): continue

            inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, val = \
                extract_nominal_split_info(tree, c, split, iterative)

            # print("S1", split, inds_l, inds_r, val, "\n")

            if(c.impurity - imp_tot <= 0):
                c.node.ttype = TTYPE_LEAF
                tree.leaves.append(c.node)
            else:
                # print("S2", split)
                ptr = _pointer_from_struct(c)
                locs = (c, split, val, iterative, node_dict, context_stack)
                node_l = new_node(locs, tree, inds_l, y_counts_l, imp_l, 0)
                node_r = new_node(locs, tree, inds_r, y_counts_r, imp_r, 1)

                split_data = SplitData(i4(split), i4(val), i4(node_l), i4(node_r), u1(False))
                #np.array([split, val, node_l, node_r, -1],dtype=np.int32)
                c.node.split_data.append(split_data)
                c.node.op_enum = OP_EQ
                # print("DONE NODE")

        # print("B")
        if(not iterative):
            SplitterContext_dtor(c)
        # print("C")
    
    assert len(tree.leaves) <= len(tree.nodes)
        # _decref_pointer(ptr)
    return 0
    # print("DONE")
    # return Tree(nodes,data_stats.u_ys)

            
# ---------------------------------------------------------------------------
# : predict()

# ----------------------------------------
# : Prediction Choice Functions

@njit(cache=True)
def get_pure_leaves(leaves):
    pure_leaves = List()
    for leaf in leaves:
        if(np.count_nonzero(leaf.counts) == 1):
            pure_leaves.append(leaf)
    return pure_leaves

pred_chooser_sig = i8(ListType(TreeNodeType))


# PROBABLY NOT EVER A GOOD CHOICE FUNCTION
@njit(pred_chooser_sig, cache=True)
def choose_majority_leaves(leaves):
    ''' If multiple leaves on predict (i.e. option tree), choose 
        the class predicted by the majority of leaves.''' 
    predictions = np.empty((len(leaves),),dtype=np.int32)
    for i,leaf in enumerate(leaves):
        predictions[i] = np.argmax(leaf.counts)
    c,u, inds = unique_counts(predictions)
    _i = np.argmax(c)
    return u[_i]

@njit(pred_chooser_sig, cache=True)
def choose_majority(leaves):
    ''' Choose the class with the largest representation in the leaves that
        select the instance being predicted. ''' 
    all_counts = leaves[0].counts.copy()
    for i in range(1, len(leaves)):
        all_counts += leaves[i].counts
    y = np.argmax(all_counts)
    return y

@njit(pred_chooser_sig, cache=True)
def choose_pure_majority(leaves):
    ''' If multiple leaves on predict (i.e. option tree), choose 
        the class predicted by the majority of putre leaves.'''
    pure_leaves = get_pure_leaves(leaves)
    leaves = pure_leaves if len(pure_leaves) > 0 else leaves
    return choose_majority(leaves)

pred_choosers = {
    "majority" : choose_majority,
    "pure_majority" : choose_pure_majority,
}

@njit(cache=True,locals={"ZERO":u1, "TO_VISIT":u1, "VISITED": u1, "_n":i4})
def filter_leaves(tree, x_nom, x_cont):
    ZERO, TO_VISIT, VISITED = 0, 1, 2
    nom_v_maps = tree.data_stats.nom_v_maps
    # Use a mask instead of a list to avoid repeats that can blow up
    #  if multiple splits are possible. Keep track of visited in case
    #  of loops (Although there should not be any loops).
    new_node_mask = np.zeros((len(tree.nodes),),dtype=np.uint8)
    new_node_mask[0] = TO_VISIT
    nodes_to_visit = np.nonzero(new_node_mask==TO_VISIT)[0]
    leaves = List()

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
                    # Determine if this sample should feed right, left, or nan (if ternary)
                    if(not sd.is_continous):
                        # Nominal case
                        mapped_val = nom_v_maps[sd.split_ind].get(x_nom[sd.split_ind],-1)
                        # if mapped_val == -1:
                        #     continue
                        _n = sd.right if(mapped_val==sd.val) else sd.left
                    else:
                        # Continous case : Need to reimplement
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
                leaves.append(node)

        #Mark all nodes_to_visit as visited so we don't mark them for a revisit
        for ind in nodes_to_visit:
            new_node_mask[ind] = VISITED

        nodes_to_visit = np.nonzero(new_node_mask==TO_VISIT)[0]
    return leaves


np_prob_item_type = np.dtype([
    ('y_class', np.int64),
    ('prob', np.float64),
])

prob_item_type = numba.from_dtype(np_prob_item_type)


@njit(cache=True)
def predict_prob(tree, X_nom, X_cont):
    '''Predicts the class associated with an unlabelled sample using a fitted 
        decision/option tree'''
    
    L = max(len(X_nom),len(X_cont))
    if(len(X_nom) == 0): X_nom = np.empty((L,0), dtype=np.int32)
    if(len(X_cont) == 0): X_cont = np.empty((L,0), dtype=np.float32)
    
    y_uvs = tree.data_stats.u_ys    
    out = np.zeros((L,len(y_uvs)),dtype=prob_item_type)
    for i in range(L):
        leaves = filter_leaves(tree, X_nom[i], X_cont[i])

        for j, y_class in enumerate(y_uvs):
            out[i][j].y_class = y_class

        for leaf in leaves:
            y = np.argmax(leaf.counts)
            out[i][y].prob += 1

        for j, y_class in enumerate(y_uvs):
            out[i][j].prob /= len(leaves)

    return out


@njit(cache=True)
def predict(tree, X_nom, X_cont):
    '''Predicts the class associated with an unlabelled sample using a fitted 
        decision/option tree'''
    
    L = max(len(X_nom),len(X_cont))
    if(len(X_nom) == 0): X_nom = np.empty((L,0), dtype=np.int32)
    if(len(X_cont) == 0): X_cont = np.empty((L,0), dtype=np.float32)
    out = np.empty((L,),dtype=np.int64)
    y_uvs = tree.data_stats.u_ys    
    for i in range(L):
        leaves = filter_leaves(tree, X_nom[i], X_cont[i])
        # for leaf in leaves:
        #     print(leaf.counts)

        # In an option tree the leaf that the instance ends up  is ambiguous 
        #   so we need a subroutine for choosing how to classify the instance 
        #   from among the leaves it is filtered into. 
        if(len(leaves) > 0):
            out_i = tree.pred_chooser(leaves)
            out_i = y_uvs[out_i]
            out[i] = out_i
        else:
            raise RuntimeError("Item does not filter into any leaves.")
            # print("BAD CASE")
            out[i] = y_uvs[0]  
    # print("OUT", out[0], y_uvs)  
    return out

# -------------------------------------------------------------------------------
# : instance ambiguity

# @njit(cache=True)
# def 



@njit(cache=True)
def get_branch_splits(tree, _node):
    # print("### NODE:", _node.index)
    covered_nodes = np.zeros(len(tree.nodes), dtype=np.uint8)
    branch_splits = Dict.empty(u8, i8)
    rec_stack = List()
    rec_stack.append(_node.index)

    while(len(rec_stack) > 0):
        node_ind = rec_stack.pop()
        node = tree.nodes[node_ind]            
        if(len(node.parents) == 0):
            continue

        for i, (p_node_ind, enc_split) in enumerate(node.parents):
            if(not covered_nodes[p_node_ind]):
                rec_stack.append(p_node_ind)
                covered_nodes[p_node_ind] = 1
            branch_splits[enc_split] = branch_splits.get(enc_split,0) + 1


    return branch_splits


@njit(cache=True)
def _count_branches(tree, _node):
    # print("### NODE:", _node.index)
    n_branches = np.zeros(len(tree.nodes),dtype=np.uint64)
    rec_stack = List()
    rec_stack.append((i4(-1), _node.index))

    while(len(rec_stack) > 0):
        prev_ind, node_ind = rec_stack.pop()
        # print(prev_ind, "->", node_ind, len(tree.nodes))

        if(n_branches[node_ind] > 0):
            # print("**", prev_ind, node_ind)
            n_branches[prev_ind] += n_branches[node_ind]
        else:
            node = tree.nodes[node_ind]            
            if(len(node.parents) == 0):
                n_branches[node_ind] = 1
                continue

            if(prev_ind > -1): rec_stack.append((prev_ind, node_ind))
            for i, (p_node_ind, enc_split) in enumerate(node.parents):
                rec_stack.append((node_ind, p_node_ind))
    return n_branches

@njit(cache=True)
def count_branches(tree, _node):
    n_branches = _count_branches(tree, _node)
    return n_branches[_node.index]

@njit(cache=True)
def _count_covering_branches(tree, _node, x_nom, x_cont):

    # Size of (N,2) for 0: not covering and 1: covering  
    n_branches = np.zeros((len(tree.nodes),2),dtype=np.uint64)
    rec_stack = List()
    rec_stack.append((i4(-1), _node.index, 1))
    nom_v_maps = tree.data_stats.nom_v_maps

    while(len(rec_stack) > 0):
        prev_ind, node_ind, was_okay = rec_stack.pop()
        # print(prev_ind, "->>", node_ind, was_okay)
        
        if(n_branches[node_ind][0] > 0 or n_branches[node_ind][1] > 0):
            if(was_okay):
                # If was_okay propogate all covering and not covering to child
                n_branches[prev_ind] += n_branches[node_ind]
            else:
                # Otherwise count all as not covering in the child
                n_branches[prev_ind][0] += np.sum(n_branches[node_ind])
        else:
            node = tree.nodes[node_ind]            
            if(len(node.parents) == 0):
                n_branches[node_ind][was_okay] = 1
                continue

            if(prev_ind > -1): rec_stack.append((prev_ind, node_ind, was_okay))

            for i, (p_node_ind, enc_split) in enumerate(node.parents):
                is_cont, negated, split, val = decode_split(enc_split)
                okay = True
                if(not is_cont):
                    mapped_val = nom_v_maps[split].get(x_nom[split],-1)
                    okay = negated ^ (mapped_val==val)
                    # print(split, "!=" if negated else "==",  val, okay)
                else:
                    pass

                rec_stack.append((node_ind, p_node_ind, okay))
    return n_branches

@njit(cache=True)
def count_covering_branches(tree, _node, x_nom, x_cont):
    n_branches = _count_covering_branches(tree, _node, x_nom, x_cont)
    out = n_branches[_node.index]
    return out[0], out[1]
        

        
        # for p_node_ind, encoded_split in leaf.parents:
        #     print(p_node_ind, decode_split(encoded_split))
        # # for s_ind in leaf.sample_inds:
            # print(tree.data_stats.X_nom[s_ind])
        

    # print(tree.context_cache)

# -------------------------------------------------------------------------------
# : str_tree()


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



def str_tree(tree, inv_mapper=None):
    '''A string representation of a tree usable for the purposes of debugging'''
    
    l = ["TREE w/ classes: %s"%tree.data_stats.u_ys]
    nom_v_inv_maps = tree.data_stats.nom_v_inv_maps
    for node in tree.nodes:
        ttype, index, splits, counts = node.ttype, node.index, node.split_data, node.counts#_unpack_node(tree,node_offset)
        op = node.op_enum
        if(ttype == TTYPE_NODE):
            s  = "NODE(%s) : " % (index)
            for i, sd in enumerate(splits):
                if(i > 0): s += "\n\t"
                if(not sd.is_continous): #<-A threshold of 1 means it's binary
                    inv_map = nom_v_inv_maps[sd.split_ind]

                    # Recover the X vector indicies and values as provided before internal remapping
                    inp_key = sd.split_ind
                    inp_val = inv_map[sd.val]

                    # If inv_mapper was provided then use it to recover the true feature key
                    #   and value before the user's vectorization preprocessing.
                    if(inv_mapper):
                        inp_key, inp_val = inv_mapper(inp_key, inp_val)

                    s += f"({inp_key},=={inp_val!r})[F:{sd.left} T:{sd.right}"
                else:
                    thresh = np.int32(sd.val).view(np.float32) if op != OP_EQ else np.int32(sd.val)
                    instr = str_op(op)+str(thresh) if op != OP_ISNAN else str_op(op)
                    s += f"({sd.split_ind},{instr})[F:{sd.left} T:{sd.right}"
                    # s += "(%s,%s)[L:%s R:%s" % (sd.split_ind,instr,sd.left,sd.right)
                s += "] "# if(split[4] == -1) else ("NaN:" + str(split[4]) + "] ")
            l.append(s)
        else:
            s  = "LEAF(%s) : %s" % (index,counts)
            l.append(s)
    return "\n".join(l)

# -------------------------------------------------------------------------------
# : Getters

@njit(cache=True)
def get_nodes(tree):
    return tree.nodes

@njit(cache=True)
def get_leaves(tree):
    return tree.leaves


tree_classifier_presets = {
    'decision_tree' : {
        'impurity_func' : 'gini',
        'split_choice' : 'single_max',
        'pred_choice' : 'majority',
        'positive_class' : 1,
        'sep_nan' : True,
        'cache_nodes' : False
    },
    'option_tree' : {
        'impurity_func' : 'gini',
        'split_choice' : 'all_max',
        'pred_choice' : 'pure_majority',
        'positive_class' : 1,
        'sep_nan' : True,
        'cache_nodes' : True
    }
}



class TreeClassifier(object):
    def __init__(self,
            preset_type='decision_tree', 
            ifit_enabled = True,
            # Optional userprovided function for mapping key value pairs back to their original
            #  values before they were vectorized 
            inv_mapper=None, 
            **kwargs):
        '''
        TODO: Finish docs
        kwargs:
            preset_type: Specifies a preset for the values of the other kwargs
            impurity_func: The name of the impurity function used 'entropy', 'gini', etc.
            split_choice: The name of the split choice policy 'all_max', etc.
            pred_choice: The prediction choice policy 'pure_majority_general' etc.
            positive_class: The integer id for the positive class (used in prediction)
            sep_nan: If set to True then use a ternary tree that treats nan's seperately 
        '''

        # If None is ever provided as config value then ignore it and use the preset value
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        kwargs = {**tree_classifier_presets[preset_type], **kwargs}

        impurity_func, split_choice, pred_choice, positive_class, sep_nan, cache_nodes = \
            itemgetter('impurity_func', 'split_choice', 'pred_choice', 'positive_class',
                'sep_nan', 'cache_nodes')(kwargs)

        self.positive_class = positive_class
        self.tree_type = self.gen_tree_type(ifit_enabled)
        self.inv_mapper = inv_mapper
        self.tree = Tree_ctor(
            self.tree_type,
            _get_wrapper_address(split_choosers[split_choice], split_chooser_sig),
            _get_wrapper_address(pred_choosers[pred_choice], pred_chooser_sig),
            _get_wrapper_address(impurity_funcs[impurity_func], impurity_func_sig),
            cache_nodes
        )

    def gen_tree_type(self, ifit_enabled):
        tf_dict = {k:v for k,v in tree_fields}
        tf = [(k,v) for k,v in {**tf_dict, **{"ifit_enabled": literal(ifit_enabled)}}.items()]
        return TreeTypeTemplate(tf)

        
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
        fit_tree(self.tree, False)
        # print("C")
    @property
    def nodes(self):
        return get_nodes(self.tree)

    @property
    def leaves(self):
        return get_leaves(self.tree)

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
        fit_tree(self.tree, True)

        


    def predict(self, X_nom, X_cont, positive_class=None):
        if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
        if(positive_class is None): positive_class = self.positive_class
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        return predict(self.tree, X_nom, X_cont)

    def predict_prob(self, X_nom, X_cont, positive_class=None):
        if(self.tree is None): raise RuntimeError("TreeClassifier must be fit before predict() is called.")
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        return predict_prob(self.tree, X_nom, X_cont)
        # return self._predict(self.tree, xb, xc, positive_class)

    def __str__(self):
        return str_tree(self.tree, self.inv_mapper)

    # def as_conditions(self,positive_class=None, only_pure_leaves=False):
    #     if(positive_class is None): positive_class = self.positive_class
    #     return tree_to_conditions(self.tree, positive_class, only_pure_leaves)



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

