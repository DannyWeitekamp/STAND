from stand.structref import define_structref, define_structref_template
from stand.utils import _struct_from_pointer, _pointer_from_struct, _pointer_from_struct_incref, _decref_pointer, _decref_structref
from numba import njit, types
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, literal, Tuple
from numba.extending import overload_method
from .data_stats import DataStatsType, DataStats_ctor, reinit_datastats
from numba.experimental.structref import new
from stand.akd import new_akd, AKDType
from stand.utils import _func_from_address
import numpy as np

#### SplitData ####

split_data_fields = [
    ('split_ind', i4),
    ('val', i4),
    ('left', i4),
    ('right', i4),
    ('is_continous', u1),
]
SplitData, SplitDataType = define_structref("SplitData", split_data_fields)
SplitDataType.__str__ = lambda self: "SplitDataType"



#### TreeNode ####

treenode_fields = [    
    ('index',i4),
    ('sample_inds', u4[::1]),
    # ('parent_index',i4),
    # ('leaf_index',i4),
    ('parents', ListType(Tuple((i4,u8)))),
    ('split_data', ListType(SplitDataType)),
    ('counts', u4[:]),
    ('conj_slip',f8),
    ('path_conj_slip',f8),
    ('ttype', u1),
    ('op_enum', u1),
    ('is_root', u1),

    ### Attributes for Hierarchical Shrinkage ###

    ('nominal_split_cache_ptrs', i8[::1]),
    # ('max_depth', i8),


    # ('y_counts_per_v', ListType(u4[:,::1])),
    # ('v_counts', ListType(u4[:,::1])),

    # # The weighted y_count contributions of the parents
    # ('par_w_y_counts_per_v', ListType(f4[:,::1])),
    # # The weighted total contributions of the parents for each value
    # ('par_v_counts', ListType(f4[::1])),
    # # How much the true counts in this split context count toward the total
    # ('self_w', f4),
]

TreeNode, TreeNodeType = define_structref("TreeNode",treenode_fields,define_constructor=False) 
TreeNodeType.__str__ = lambda self: "TreeNodeType"


OP_NOP = u1(0)
OP_GE = u1(1)
OP_LT = u1(2) 
OP_ISNAN = u1(3)
OP_EQ = u1(4)

# i4_arr = i4[:]

i4_u8_tup_type = Tuple((i4,u8))

@njit(cache=True)
def TreeNode_ctor(ttype, index, sample_inds, counts, tree, is_root=False):
    st = new(TreeNodeType)
    st.index = index
    st.sample_inds = sample_inds    
    st.split_data = List.empty_list(SplitDataType)
    st.parents = List.empty_list(i4_u8_tup_type)
    st.counts = counts
    st.conj_slip = 0.0
    st.path_conj_slip = 0.0
    st.ttype = ttype
    st.op_enum = OP_NOP
    st.nominal_split_cache_ptrs = np.zeros(tree.data_stats.X_nom.shape[1], dtype=np.int64)
    st.is_root = is_root
    return st


#### SplitterContext ####

splitter_context_fields = [
    
    #A pointer to the parent split context
    # ('parent_ptr', i8),

    ('split_chain', u8[::1]),

    ### Attributes inserted at initialization ###

    # The node in the output tree associated with this context
    ('node', TreeNodeType),

    ('root_context_ptr', i8),
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
SplitterContextType.__str__ = lambda self: "SplitterContextType"
SplitterContext.__str__ = lambda self: f"<SplitterContext at {hex(id(self))}>"


@njit(cache=True)
def SplitterContext_ctor(split_chain):
    st = new(SplitterContextType)
    st.n_last_update = 0 
    st.nominal_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    st.continous_split_cache_ptrs = np.zeros((32,),dtype=np.int64)
    return st

@njit(cache=True)
def reinit_splittercontext(c, node, root_c, sample_inds, y_counts, impurity):
    c.node = node
    if(root_c is None):
        c.root_context_ptr = _pointer_from_struct_incref(c)
    else:
        c.root_context_ptr = _pointer_from_struct_incref(root_c)

    c.sample_inds = sample_inds
    c.y_counts = y_counts
    c.impurity = impurity
    c.best_split_impurity = np.inf


@njit(cache=True)
def SplitterContext_dtor(sc):
    for ptr in sc.nominal_split_cache_ptrs:
        if(ptr != 0):
            _decref_pointer(ptr)

#### Tree Params ####

tree_params_fields = [
    # How much slip 
    ('slip', f8), 

    ('n_slip_atten', f8), 

    # Weight slip leaves by path slip
    ('w_path_slip', types.boolean), 

    # A list of the actual nodes of the tree.
    # Regularization terms for hierarchical shrink
    ('lam_p', f8), # Probabilities
    ('lam_e', f8), # Specific Extensions
    ('lam_l', f8), # Leaves 

   
]

TreeParams, TreeParamsType = define_structref("TreeParams", tree_params_fields, define_constructor=True)


#### Tree ####

i8_arr = i8[::1]
# impurity_func_sig = f8(f4[:], i4[:], i4)
impurity_func_sig = f8(f4[:])
split_chooser_sig = Tuple((i8[::1],f8[::1]))(TreeParamsType, f8[::1], i8)
pred_chooser_sig = i8(ListType(TreeNodeType))



tree_fields = [
    # A list of the actual nodes of the tree.
    ('nodes', ListType(TreeNodeType)),
    ('roots', ListType(TreeNodeType)),
    ('leaves', ListType(TreeNodeType)),

    # A cache of split contexts keyed by the sequence of splits so far
    #  this is where split statistics are held between calls to ifit().
    ('context_cache', AKDType(u8[::1], SplitterContextType)),

    # The data stats for this tree. This is kept around be between calls 
    #  to ifit() and replaced with each call to fit().
    ('data_stats', DataStatsType),

    # Decides which feature(s) to split on based on an array of impurities.
    ('split_chooser', types.FunctionType(split_chooser_sig)),

    # Decides which class to predict based on a list of leaves that an example falls into.
    ('pred_chooser', types.FunctionType(pred_chooser_sig)),

    # Calculates the impurity of a distribution of classes selected by a node.
    ('impurity_func', types.FunctionType(impurity_func_sig)),
    
    
    ('params', TreeParamsType),

    # The positive class (only relevant for sequential covering)
    ('pos_y', i4),

    # Whether or not nodes should be cached
    ('cache_nodes', types.boolean),    

    # Whether or not iterative fitting is enabled
    # ('ifit_enabled', literal(True)),
    ('ifit_enabled', literal(True)),
]

Tree, TreeTypeTemplate = define_structref_template("Tree", tree_fields, define_constructor=False)


u8_arr = u8[::1]



impurity_func_type = types.FunctionType(impurity_func_sig)
split_chooser_type = types.FunctionType(split_chooser_sig)
pred_chooser_type = types.FunctionType(pred_chooser_sig)


MIN_i4 = -2147483648

@njit(cache=True)
def Tree_ctor(tree_type, split_chooser_addr, pred_chooser_addr,
         impurity_func_addr, cache_nodes, pos_y,
         # Tree Params
         slip, n_slip_atten, w_path_slip, lam_p, lam_l, lam_e):
    st = new(tree_type)
    st.nodes = List.empty_list(TreeNodeType)
    st.roots = List.empty_list(TreeNodeType)
    st.leaves = List.empty_list(TreeNodeType)
    # st.u_ys = np.zeros(0,dtype=np.int32)
    st.context_cache = new_akd(u8_arr,SplitterContextType)#Dict.empty(i8_arr, SplitterContextType)
    st.pos_y = pos_y
    ds = st.data_stats = DataStats_ctor(pos_y)


    st.impurity_func = _func_from_address(impurity_func_type, impurity_func_addr)
    st.split_chooser = _func_from_address(split_chooser_type, split_chooser_addr)
    st.pred_chooser = _func_from_address(pred_chooser_type, pred_chooser_addr)
    
    par = st.params = TreeParams(slip, float(n_slip_atten), w_path_slip, lam_p, lam_l, lam_e)
 


    # par.slip = slip 
    # par.n_slip_atten = n_slip_atten 
    # par.w_path_slip = w_path_slip 
    
    # par.lam_p = lam_p   
    # par.lam_l = lam_l   
    # par.lam_e = lam_e   
    
    st.cache_nodes = cache_nodes 
    
    return st
    
@njit(cache=True)
def reinit_tree_datastats(tree, X_nom, X_cont, Y):
    ds = tree.data_stats = DataStats_ctor(tree.pos_y)
    reinit_datastats(ds, X_nom, X_cont, Y)


