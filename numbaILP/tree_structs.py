from numbaILP.structref import define_structref, define_structref_template
from numbaILP.utils import _struct_from_pointer, _pointer_from_struct, _pointer_from_struct_incref, _decref_pointer, _decref_structref
from numba import njit, types
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, literal
from .data_stats import DataStatsType, DataStats_ctor, reinit_datastats
from numba.experimental.structref import new
from numbaILP.akd import new_akd, AKDType
import numpy as np

#### SplitData ####

split_data_fields = [
    ('is_continous', u1),
    ('split_ind', i4),
    ('val', i4),
    ('left', i4),
    ('right', i4)
]
SplitData, SplitDataType = define_structref("SplitData", split_data_fields)
SplitDataType.__str__ = lambda self: "SplitDataType"


#### TreeNode ####

treenode_fields = [
    ('ttype',u1),
    ('index',i4),
    ('op_enum', u1),
    ('split_data',ListType(SplitDataType)),
    ('counts', u4[:])
]

TreeNode, TreeNodeType = define_structref("TreeNode",treenode_fields,define_constructor=False) 
TreeNodeType.__str__ = lambda self: "TreeNodeType"


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
    

    #A pointer to the parent split context
    # ('parent_ptr', i8),

    ('split_chain', u8[::1]),

    ### Attributes inserted at initialization ###

    # The node in the output tree associated with this context
    ('node', TreeNodeType),
    # The indicies of all samples that filter into this node
    ('sample_inds', u4[::1]),
    # The counts of each class label in this node
    ('y_counts',u4[:]),    
    # The impurity of the node before splitting
    ('impurity', f8),
    
    ### Attributes resolved after impurities computed ###

    # The total, left, and right impurities of all the splits f8[n_features,3]
    ('impurities', f8[:,:]),
    # The impurity of the node after the best split
    ('best_split_impurity', f8),
    # Whether the best split is nominal 0 or continuous 1
    ('best_is_continous', u1),
    # The index of the best split 
    ('best_split', i4),
    # In the nominal case the value of the best literal
    ('best_val', i4),
    # In the continous case the value of the best threshold
    ('best_thresh', f4),

    ### Data meant to be reused between calls to ifit() ### 

    #The time of the most recent update to this context
    ('n_last_update', i4),
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
def SplitterContext_ctor(split_chain):
    st = new(SplitterContextType)    

    st.split_chain = split_chain
    st.n_last_update = 0 
    st.nominal_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    st.continous_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    
    return st

@njit(cache=True)
def reinit_splittercontext(c, node, sample_inds, y_counts, impurity):
    c.node = node
    c.sample_inds = sample_inds
    c.y_counts = y_counts
    c.impurity = impurity

    c.best_split_impurity = np.inf





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
    # ('u_ys', i4[::1]),

    # A cache of split contexts keyed by the sequence of splits so far
    #  this is where split statistics are held between calls to ifit().
    ('context_cache', AKDType(u8[::1], SplitterContextType)),

    # The data stats for this tree. This is kept around be between calls 
    #  to ifit() and replaced with each call to fit().
    ('data_stats', DataStatsType),

    # Whether or not iterative fitting is enabled
    ('ifit_enabled', literal(False)),

    ('split_chooser', types.FunctionType(i8[::1](f8[::1])))

]



Tree, TreeTypeTemplate = define_structref_template("Tree", tree_fields, define_constructor=False)

u8_arr = u8[::1]
@njit(cache=True)
def Tree_ctor(tree_type, split_chooser):
    st = new(tree_type)
    st.nodes = List.empty_list(TreeNodeType)
    # st.u_ys = np.zeros(0,dtype=np.int32)
    st.context_cache = new_akd(u8_arr,SplitterContextType)#Dict.empty(i8_arr, SplitterContextType)
    st.data_stats = DataStats_ctor()
    st.split_chooser = split_chooser
    return st
    
@njit(cache=True)
def reinit_tree_datastats(tree, X_nom, X_cont, Y):
    ds = tree.data_stats = DataStats_ctor()
    reinit_datastats(ds, X_nom, X_cont, Y)


