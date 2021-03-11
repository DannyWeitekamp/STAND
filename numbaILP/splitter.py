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
from numba.np.ufunc.parallel import _get_thread_id
from sklearn.preprocessing import OneHotEncoder
from numbaILP.fnvhash import hasharray#, AKD#, akd_insert,akd_get

config.THREADING_LAYER = 'thread_safe'
print("n threads", config.NUMBA_NUM_THREADS)
# os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '1'


CRITERION_none = 0
CRITERION_gini = 1
CRITERION_entropy = 2

SPLIT_CHOICE_single_max = 1
SPLIT_CHOICE_all_max = 2

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



nominal_split_cache_fields = [
    ('best_v', i4),
    ('v_counts', u4[:]),
    ('y_counts_per_v', u4[:,:]),
]

NominalSplitCache, NominalSplitCacheType = define_structref("NominalSplitCache",nominal_split_cache_fields, define_constructor=False) 

@njit(cache=True)
def NominalSplitCache_ctor(n_vals, n_classes):
    st = new(NominalSplitCacheType)
    # instantiate as one so it is gaurenteed to be contiguous
    # data = np.zeros((n_vals*(n_classes+1)))
    # st.v_counts = data[:n_vals]
    # st.y_counts_per_v = data[n_vals:n_vals*(n_classes+1)].reshape((n_vals,n_classes))
    st.best_v = -1
    st.v_counts = np.zeros((n_vals,),dtype=np.uint32)
    st.y_counts_per_v = np.zeros((n_vals,n_classes),dtype=np.uint32)
    return st

@njit(cache=True)
def expand_nominal_split_cache(st, n_vals,n_classes):
    v_counts = np.empty((n_vals,),dtype=np.uint32)
    v_counts[:len(st.v_counts)] = st.v_counts
    v_counts[len(st.v_counts):] = 0
    st.v_counts = v_counts

    y_counts_per_v = np.empty((n_vals,n_classes),dtype=np.uint32)
    shp = st.y_counts_per_v.shape
    y_counts_per_v[:shp[0],:shp[1]] = st.y_counts_per_v
    y_counts_per_v[:shp[0],shp[1]:] = 0
    y_counts_per_v[:shp[0]] = 0
    st.y_counts_per_v = y_counts_per_v
    return st


split_data_fields = [
    ('is_continous', u1),
    ('split_ind', i4),
    ('val', i4),
    ('left', i4),
    ('right', i4)
]
SplitData, SplitDataType = define_structref("SplitData", split_data_fields)

treenode_fields = [
    ('ttype',u1),
    ('index',i4),
    ('op_enum', u1),
    ('split_data',ListType(SplitDataType)),
    ('counts', u4[:])
]

TreeNode, TN = define_structref("TreeNode",treenode_fields,define_constructor=False) 


OP_NOP = u1(0)
OP_GE = u1(1)
OP_LT = u1(2) 
OP_ISNAN = u1(3)
OP_EQ = u1(4)

# i4_arr = i4[:]

@njit(cache=True)
def TreeNode_ctor(ttype, index, counts):
    st = new(TN)
    st.ttype = ttype
    st.index = index
    st.op_enum = OP_NOP
    st.split_data = List.empty_list(SplitDataType)
    st.counts = counts
    return st





continous_split_cache_field = [
    ('is_const', u1),
    ('threshold', f8),
    ('op', i4),
    ('left_counts', i4[:]),
    ('right_counts', i4[:]),
    ('nan_counts', i4[:]),
]

ContinousSplitCache, ContinousSplitCacheType = define_structref("ContinousSplitCache",continous_split_cache_field, define_constructor=False) 

# counts_imps = np.dtype([('left_count', np.float64), ('col', np.float64)])

splitter_context_fields = [
    #The time of the most recent update to this context
    ('t_last_update', i4),

    #A pointer to the parent split context
    ('parent_ptr', i8),
    ('node', TN),

    

    
    
    # Idea borrowed from sklearn, sample_inds are preallocated and 
    #  kept contiguous in each node by swapping left and right indicies 
    #  then only 'start' and 'end' need to be passed instead of copying the indicies
    ('sample_inds', u4[::1]),
    # ('start',i8),
    # ('end', i8),

    

    # The counts of each class label in this node
    ('y_counts',u4[:]),
    # The number of unique class labels 
    ('n_classes',i4),
    
    #The impurity of the node before splitting
    ('impurity', f8),
    #The total, left, and right impurities of all the splits f8[n_features,3]
    ('impurities', f8[:,:]),
    #The impurity of the node after the best split
    ('best_split_impurity', f8),

    #Whether the best split is nominal 0 or continuous 1
    ('best_is_continous', u1),
    ('best_split', i4),
    

    # In the nominal case the value of the best literal
    ('best_val', i4),
    # In the continous case the value of the best threshold
    ('best_thresh', f4),


    # Whether or not the y_counts associated with selecting on each nominal
    #  value are cached
    ('nominal_split_cache_ptrs', i8[:]),
    ('continous_split_cache_ptrs', i8[:]),
    # ('val_y_counts_cached', u1),
    # # Whether or not the left and right y_counts are cached 
    # ('split_y_counts_cached', u1),
    
    # A raw data array that holds for each feature:
    #  v_count: u4[n_vals_j]
    #  y_counts : u4[n_vals_j, n_classes]
    # ('val_vy_count_caches', u4[:]),
    # ???
    # ('split_y_count_caches', u4[:]),
    # A cache of the best thresholds for each continous split
    # ('threshold_cache', f4[:]),

]

SplitterContext, SplitterContextType = define_structref("SplitterContext",splitter_context_fields, define_constructor=False) 

data_stats_fields = [
    #The feature matrix
    ('X', i4[:,:]),
    #The label list
    ('Y', i4[:]),
    # The number of unique values per nominal feature 
    ('n_vals', i4[::1]),
    # ('feature_inds', i8[::1]),
    # The total number of samples for this node
    ('n_samples', i4),
    ('n_classes', i4),
    ('u_ys', i4[::1]),
    ('y_counts', u4[::1]),
    # # The number of constant features
    # ('n_const_fts', i8),
    # The total number of features
    ('n_features', i4),
]

DataStats, DataStatsType = define_structref("DataStats",data_stats_fields, define_constructor=False) 

@njit(cache=True)
def DataStats_ctor(X, Y):
    st = new(DataStatsType)

    y_counts,u, inds = unique_counts(Y)
    st.X = X
    st.Y = inds
    
    

    # for x in X:
    #     x_sorted = np.sort(x)
    #     y_counts,u, inds = unique_counts(Y)


    n_vals = np.empty(X.shape[1],dtype=np.int32)

    for j in range(X.shape[1]):
        n_vals[j] = np.max(X[:,j])+1



    st.n_classes = len(u)
    st.u_ys = u
    st.y_counts = y_counts

    st.n_vals = n_vals#(np.ones((X.shape[1],))*5).astype(np.int32)
    st.n_samples = X.shape[0]
    st.n_features = X.shape[1]
    return st


@njit(cache=True)
def SplitterContext_ctor(parent_ptr, node, sample_inds, y_counts, impurity):
    st = new(SplitterContextType)
    # st.counts_cached = False
    st.parent_ptr = parent_ptr
    st.node = node
    st.sample_inds = sample_inds
    # st.start = start
    # st.end = end
    # st.n_samples = len(sample_inds)
    # st.n_samples = end-start

    # st.n_classes = n_classes
    st.y_counts = y_counts

    st.impurity = impurity
    st.best_split_impurity = np.inf

    st.nominal_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    st.continous_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    # st.counts_imps = np.zeros(n_features, ((n_classes)*2)+6,dtype=np.int32)
    # if(parent_ptr != 0):
    #     parent = _struct_from_pointer(SplitterContextType, parent_ptr)
    #     st.n_vals = parent.n_vals
    #     # st.sample_inds = parent.sample_inds
    #     st.n_classes = parent.n_classes
    #     st.n_features = parent.n_features
    #     # st.feature_inds = parent.feature_inds
    #     st.X = parent.X
    #     st.Y = parent.Y
        # st.n_vals = parent.n_vals
    return st


@njit(cache=True)
def SplitterContext_dtor(sc):
    for ptr in sc.nominal_split_cache_ptrs:
        if(ptr != 0):
            _decref_pointer(ptr)

@njit(cache=True,parallel=False)
def update_nominal_impurities(data_stats, splitter_context):
    #Make various variables local
    ds = data_stats
    sc = splitter_context

    # n_samples, start, end = sc.n_samples, sc.start, sc.end
    X, Y = ds.X, ds.Y
    n_vals = ds.n_vals
    # n_samples = ds.n_samples
    n_classes = ds.n_classes
    
    # counts_imps = sc.counts_imps
    
    # feature_inds = sc.feature_inds
    # n_const_fts = sc.n_const_fts
    sample_inds = sc.sample_inds
    n_samples = len(sc.sample_inds)
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
    impurities = np.empty((X.shape[1],3),dtype=np.float64)
    # b_split, b_split_imp_total = 0, np.inf
    #Go through the samples in Fortran order (i.e. feature then sample)
    # for k_j in prange(0,n_non_const):
    for k_j in prange(X.shape[1]):
        # print(_get_thread_id(),k_j)
        j = k_j#feature_inds[k_j]
        n_vals_j = n_vals[j]
        cache_ptr = sc.nominal_split_cache_ptrs[j]

        if(cache_ptr != 0):
            split_cache = _struct_from_pointer(NominalSplitCacheType, cache_ptr)
            split_cache_shape = split_cache.y_counts_per_v.shape
            if(split_cache_shape[0] != n_vals_j or split_cache_shape[1] != n_classes):
                expand_nominal_split_cache(split_cache, n_vals_j, n_classes)
        else:
            split_cache = NominalSplitCache_ctor(n_vals_j, n_classes)
            sc.nominal_split_cache_ptrs[j] = _pointer_from_struct_incref(split_cache)

        # print(cache_ptr,sc.nominal_split_cache_ptrs[j])

        # print("ZB")
        v_counts       = split_cache.v_counts
        y_counts_per_v = split_cache.y_counts_per_v

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
        

        #If this feature is found to be constant then skip computing impurity
        if(np.sum(v_counts > 0) <= 1):
            split_cache.best_v = 0
            impurities[k_j,0] = impurity
            impurities[k_j,1] = impurity
            impurities[k_j,2] = impurity
        else:
            b_imp_tot, b_imp_l, b_imp_r, b_ft_val = np.inf, 0, 0, 0
            for ft_val in range(n_vals_j):
                counts_r = y_counts_per_v[ft_val]
                total_r = np.sum(counts_r)

                counts_l = y_counts-counts_r
                total_l = n_samples-total_r

                imp_l = gini(total_l, counts_l)
                imp_r = gini(total_r, counts_r)

                imp_tot = ((total_l/n_samples) * imp_l) + ((total_r/n_samples) * imp_r)
                if(imp_tot < b_imp_tot):
                    b_imp_tot, b_imp_l, b_imp_r, b_ft_val = imp_tot, imp_l, imp_r, ft_val
            split_cache.best_v = b_ft_val
            impurities[k_j,0] = b_imp_tot
            impurities[k_j,1] = b_imp_l
            impurities[k_j,2] = b_imp_r

    sc.impurities = impurities
    # print(impurities)
    # print("ZC")
    # print(best_ind)


@njit(cache=True)
def build_root(X_nom,x_cont,Y):
    sorted_inds = np.argsort(Y)
    # X = np.asfortranarray(X[sorted_inds])
    X = X_nom[sorted_inds]
    Y = Y[sorted_inds]

    
    sample_inds = np.arange(len(Y),dtype=np.uint32)

    ds = DataStats_ctor(X,Y)

    impurity = gini(len(Y),ds.y_counts)
    

    node_dict = Dict.empty(u4,BE_List)
    nodes = List.empty_list(TN)
    node = TreeNode_ctor(TTYPE_NODE,i4(0),ds.y_counts)
    nodes.append(node)

    c = SplitterContext_ctor(0, node, sample_inds , ds.y_counts, impurity)
    context_stack = List.empty_list(SplitterContextType)
    context_stack.append(c)



    return ds, context_stack, node_dict, nodes


@njit(cache=True)
def choose_next_splits():
    pass


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


TTYPE_NODE = u1(1)
TTYPE_LEAF = u1(2)

@njit(cache=True)
def new_node(locs, c_ptr, sample_inds,y_counts,impurity):
    node_dict, nodes, context_stack, cache_nodes = locs
        # node_dict,nodes,new_contexts,cache_nodes = locs
        # NODE, LEAF = i4(1), i4(2) #np.array(1,dtype=np.int32).item(), np.array(2,dtype=np.int32).item()
    node_id = i4(-1)
    if (cache_nodes): node_id= akd_get(node_dict,sample_inds)
    if(node_id == -1):
        node_id = i4(len(nodes))
        if(cache_nodes): akd_insert(node_dict,sample_inds,node_id)
        if(impurity > 0.0):
            node = TreeNode_ctor(TTYPE_NODE,node_id,y_counts)
            nodes.append(node)
            new_c = SplitterContext_ctor(c_ptr, node, sample_inds, y_counts, impurity)
            context_stack.append(new_c)
            # new_contexts.append(SplitContext(new_inds,
            #     impurity,countsPS[split,ind], node))
        else:
            nodes.append(TreeNode_ctor(TTYPE_LEAF,node_id,y_counts))
    return node_id


Tree, TreeType = define_structref("Tree",[("nodes",ListType(TN)),('u_ys', i4[::1])])

@njit(cache=True)
def extract_nominal_split_info(data_stats, c, split):
    bst_imps = c.impurities[split]
    imp_tot, imp_l, imp_r = bst_imps[0], bst_imps[1], bst_imps[2]
    # print("Q")
    splt_c = _struct_from_pointer(NominalSplitCacheType, c.nominal_split_cache_ptrs[split])
    best_v = splt_c.best_v
    # print(splt_c.y_counts_per_v)
    # print(splt_c.best_v)
    y_counts_r = splt_c.y_counts_per_v[splt_c.best_v]
    y_counts_l = c.y_counts - y_counts_r
    # print("P")
    # print(c.y_counts,y_counts_l, y_counts_r)

    inds_l = np.empty(np.sum(y_counts_l), dtype=np.uint32)
    inds_r = np.empty(np.sum(y_counts_r), dtype=np.uint32)
    p_l, p_r = 0, 0
    for ind in c.sample_inds:
        if (data_stats.X[ind, split]==splt_c.best_v):
            inds_r[p_r] = ind
            p_r += 1
        else:
            inds_l[p_l] = ind
            p_l += 1

    # print(inds_l, inds_r)
    # print("P")
    # print(inds_l, inds_r)

    return inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, best_v
            



@njit(cache=True, locals={'y_counts_l' : u4[:], 'y_counts_r' : u4[:]})
def fit_tree(X_nom, X_cont, Y, config,iterative=False):
    '''
        X : ndarray of i4, needs to start at 0 and 
    '''
    cache_nodes = False
    # print("Z")
    data_stats, context_stack, node_dict, nodes = build_root(X_nom, X_cont,Y)
    
    # print("A")
    

    while(len(context_stack) > 0):
        # print("A")
        c = context_stack.pop()
        update_nominal_impurities(data_stats,c)
        # print(c.impurities[:,0],c.start,c.end)
        best_split = np.argmin(c.impurities[:,0])
        for split in [best_split]:
            # print("S", split)

            inds_l, inds_r, y_counts_l, y_counts_r, imp_tot, imp_l, imp_r, val = \
                extract_nominal_split_info(data_stats, c, split)

            # print("S1", split)

            # if(impurity_decrease[split] <= 0.0):
            

            if(c.impurity - imp_tot <= 0):
                c.node.ttype = TTYPE_LEAF

            else:
                # print("S2", split)
                ptr = _pointer_from_struct(c)
                locs = (node_dict, nodes, context_stack, cache_nodes)
                node_l = new_node(locs, ptr, inds_l, y_counts_l, imp_l)
                node_r = new_node(locs, ptr, inds_r, y_counts_r, imp_r)

                split_data = SplitData(u1(False),i4(split), i4(val), i4(node_l), i4(node_r))
                #np.array([split, val, node_l, node_r, -1],dtype=np.int32)
                c.node.split_data.append(split_data)
                c.node.op_enum = OP_EQ

            # print("B")
            SplitterContext_dtor(c)
            # print("C")
        
        # _decref_pointer(ptr)
        
    # print("DONE")
    return Tree(nodes,data_stats.u_ys)

            

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

@njit(nogil=True,fastmath=True, cache=True, locals={"ZERO":u1, "VISIT":u1, "VISITED": u1, "_n":i4})
def predict_tree(tree, x_nom, x_cont, config):#, pred_choice_enum, positive_class=0,decode_classes=True):
    '''Predicts the class associated with an unlabelled sample using a fitted 
        decision/ambiguity tree'''
    ZERO, VISIT, VISITED = 0, 1, 2
    L = max(len(x_nom),len(x_cont))
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
                if(node.ttype == TTYPE_NODE):
                    # Test every split in the node. Again in a traditional decision tree
                    #  there should only be one split per node.
                    for sd in node.split_data:
                        # split_on, ithresh, left, right, nan  = s[0],s[1],s[2],s[3],s[4]

                        # Determine if this sample should feed right, left, or nan (if ternary)
                        if(not sd.is_continous):
                            # Nominal case
                            if(x_nom[i,sd.split_ind]==sd.val):
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
                        if(new_node_mask[_n] != VISITED): new_node_mask[_n] = VISIT
                            
                else:
                    leafs.append(node.counts)

            node_inds = np.nonzero(new_node_mask==VISIT)[0]
        # print(leafs)
        # Since the leaf that the sample ends up in is ambiguous for an ambiguity
        #   tree we need a subroutine that chooses how to classify the sample from the
        #   various leaves that it could end up in. 
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
    l = ["TREE w/ classes: %s"%tree.u_ys]
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
                    s += f"({sd.split_ind},=={sd.val})[L:{sd.left} R:{sd.right}"
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
        # total_enum = g.get(f"TOTAL_{total_func}",None)
        split_choice_enum = g.get(f"SPLIT_CHOICE_{split_choice}",None)
        pred_choice_enum = g.get(f"PRED_CHOICE_{pred_choice}",None)

        if(criterion_enum is None): raise ValueError(f"Invalid criterion {criterion}")
        # if(total_enum is None): raise ValueError(f"Invalid criterion {total_func}")
        if(split_choice_enum is None): raise ValueError(f"Invalid split_choice {split_choice}")
        if(pred_choice_enum is None): raise ValueError(f"Invalid pred_choice {pred_choice}")
        self.positive_class = positive_class

        config_dict = {k:v for k,v in config_fields}
        config_dict['criterion_enum'] = literal(criterion_enum)
        config_dict['split_choice_enum'] = literal(split_choice_enum)
        config_dict['pred_choice_enum'] = literal(pred_choice_enum)

        ConfigType = TreeClassifierConfigTemplate([(k,v) for k,v in config_dict.items()])

        # print(config_dict)

        # self.config = 

        

        self.config = new_config(ConfigType)
        self.tree = None

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
        
        
    def fit(self,X_nom,X_cont,Y,miss_mask=None, ft_weights=None):
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        # if(miss_mask is None): miss_mask = np.zeros_like(xc, dtype=np.bool)
        # if(ft_weights is None): ft_weights = np.empty(xb.shape[1]+xc.shape[1], dtype=np.float64)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        Y = Y.astype(np.int32)
        # miss_mask = miss_mask.astype(np.bool)
        # ft_weights = ft_weights.astype(np.float64)
        # assert miss_mask.shape == xc.shape

        # self.tree = self._fit(xb, xc, y, miss_mask, ft_weights)
        self.tree = fit_tree(X_nom, X_cont, Y, self.config, False)

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

TreeClassifier()

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
    dt = TreeClassifier()
    dt.fit(X, X_cont, Y)
    # fit_tree(X, X_cont, Y)
    

def test_sklearn():
    clf = SKTree.DecisionTreeClassifier()
    clf.fit(X_oh,Y)



# print(time_ms(test_fit_tree))
# print(time_ms(test_sklearn))







dt = TreeClassifier()
dt.fit(X, X_cont, Y)
print(dt)
# print_tree(tree)


# print("PRINT")

#### test_basics ####

def setup1():
    data1 = np.asarray([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
    [0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
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

# print("PRINT")

X,Y = setup1()
dt.fit(X, X_cont, Y)
print(dt)
print(dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))

X,Y = setup2()
dt.fit(X, X_cont, Y)
print(dt)
print(dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))

X,Y = setup3()
dt.fit(X, X_cont, Y)
print(dt)
print(dt.predict(X, X_cont)) #predict_tree(tree,X, X_cont, PRED_CHOICE_majority))


                


            # counts_imps[0+c+y_j] += 1













# class DecisionTree2(TreeClassifier):
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


dt2 = DecisionTree2()

dt2.ifit({'A': 0, 'B' : 0},0)
dt2.ifit({'A': 0, 'B' : 1},0)
dt2.ifit({'A': 1, 'B' : 0},0)
dt2.ifit({'A': 1, 'B' : 1},1)


print(dt2.predict([{'A': 0, 'B' : 0}]))
print(dt2.predict([{'A': 0, 'B' : 1}]))
print(dt2.predict([{'A': 1, 'B' : 0}]))
print(dt2.predict([{'A': 1, 'B' : 1}]))

print(dt2.dt)


