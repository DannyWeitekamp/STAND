from numbaILP.structref import define_structref, define_structref_template
from numba import njit
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type


#### SplitData ####

split_data_fields = [
    ('is_continous', u1),
    ('split_ind', i4),
    ('val', i4),
    ('left', i4),
    ('right', i4)
]
SplitData, SplitDataType = define_structref("SplitData", split_data_fields)


#### TreeNode ####

treenode_fields = [
    ('ttype',u1),
    ('index',i4),
    ('op_enum', u1),
    ('split_data',ListType(SplitDataType)),
    ('counts', u4[:])
]

TreeNode, TreeNodeType = define_structref("TreeNode",treenode_fields,define_constructor=False) 


OP_NOP = u1(0)
OP_GE = u1(1)
OP_LT = u1(2) 
OP_ISNAN = u1(3)
OP_EQ = u1(4)

# i4_arr = i4[:]

@njit(cache=True)
def TreeNode_ctor(ttype, index, counts):
    st = new(TreeNodeType)
    st.ttype = ttype
    st.index = index
    st.op_enum = OP_NOP
    st.split_data = List.empty_list(SplitDataType)
    st.counts = counts
    return st





#### SplitterContext ####

splitter_context_fields = [
    #The time of the most recent update to this context
    ('t_last_update', i4),

    #A pointer to the parent split context
    ('parent_ptr', i8),
    ('node', TreeNodeType),

    
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




#### Tree ####

i8_arr = i8[::1]

tree_fields = [

    # A list of the actual nodes of the tree.
    ('nodes',ListType(TreeNodeType)),
    ('u_ys', i4[::1]),

    # A cache of split contexts keyed by the sequence of splits so far
    #  this is where split statistics are held between runs.
    ('context_cache', DictType(i8[::1],SplitContext)),

    # The data stats for this tree. This is kept around be between iterative fits.
    ('data_stats', optional(DataStatsType)),

    # Whether or not iterative fitting is enabled
    ('ifit_enabled', literal(False)),

]



Tree, TreeTypeTemplate = define_structref_template("Tree", tree_fields, define_constructor=False)

@njit(cache=True)
def Tree_ctor(tree_type):
    st = new(tree_type)
    st.nodes = List.empty_list(TreeNodeType)
    st.u_ys = np.zeros(0,dtype=np.int32)
    st.context_cache = Dict.empty(i8_arr, SplitterContext)
    st.data_stats = None
    
