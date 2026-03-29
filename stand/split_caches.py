from stand.structref import define_structref, define_structref_template
from stand.utils import _struct_from_pointer, _pointer_from_struct, _pointer_from_struct_incref, _decref_pointer, _decref_structref
from numba import njit
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type
from numba.experimental.structref import new
import numpy as np


''' A split cache is responsible for storing information about candidate splits
    at particular nodes in the tree. If ifit_enabled=True then these are stored
    in the context_cache of the the tree. 
''' 

#### NominalSplitCache ####

nominal_split_cache_fields = [
    ### Core Cache Info ###

    # The best value to split on
    ('best_v', i4),
    # The counts for each value
    ('v_counts', u4[:]),
    # The counts for each label by selecting on each value
    ('y_counts_per_v', u4[:,:]),

    ### Cache info for left and right indicies ###

    # The best value to split on the previous time this was updated
    ('prev_best_v', i4),
    # Cache of the indicies for the samples going into the left and right 
    ('l_inds', u4[::1]),
    ('r_inds', u4[::1]),
    # Buffer for l_inds and r_inds to help reduce frequency of resizing
    ('l_inds_buffer', u4[::1]),
    ('r_inds_buffer', u4[::1]),

    # The number of samples in the previous update
    ('n_last_update', i4),

    ### Attributes for Hierarchical Shrinkage ###

    # The weighted y_count contributions of the parents
    ('par_w_y_probs_per_v', f4[:,:]),
    # # The weighted total contributions of the parents for each value
    # ('par_w_v_probs', f4[:]),

    # The total weighted y_count contributions
    ('w_y_probs_per_v', f4[:,:]),
    # The weighted total contributions of the parents for each value
    # ('w_v_probs', f4[:]),
    # How much the true counts in this split context count toward the total
    # ('self_w', f4),


    # The number of samples that went to the left and right in the previous
    #  update
    # ('n_l_last_update', i4),
    # ('n_r_last_update', i4),
    
]

NominalSplitCache, NominalSplitCacheType = \
    define_structref("NominalSplitCache", nominal_split_cache_fields,
        define_constructor=False) 

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

    st.prev_best_v = -1
    st.n_last_update = 0
    # st.n_l_last_update = 0
    # st.n_r_last_update = 0
    return st

@njit(cache=True)
def expand_nominal_split_cache(st, n_vals,n_classes):
    v_counts = np.empty((n_vals,),dtype=np.uint32)
    v_counts[:len(st.v_counts)] = st.v_counts
    v_counts[len(st.v_counts):] = 0
    st.v_counts = v_counts


    # print("EXPAND", st.y_counts_per_v)
    y_counts_per_v = np.empty((n_vals,n_classes),dtype=np.uint32)
    shp = st.y_counts_per_v.shape
    y_counts_per_v[:shp[0],:shp[1]] = st.y_counts_per_v
    y_counts_per_v[:shp[0],shp[1]:] = 0
    y_counts_per_v[shp[0]:] = 0
    st.y_counts_per_v = y_counts_per_v
    # print("AFTER", st.y_counts_per_v)
    return st



#### ContinuousSplitCache ####

continuous_split_cache_fields = [
    # The best threshold found (split point: left < best_thresh, right >= best_thresh)
    ('best_thresh', f4),
    # Label counts for samples going left (X_cont[i,j] < best_thresh)
    ('y_counts_l', u4[:]),
    # Label counts for samples going right (X_cont[i,j] >= best_thresh)
    ('y_counts_r', u4[:]),
]

ContinuousSplitCache, ContinuousSplitCacheType = \
    define_structref("ContinuousSplitCache", continuous_split_cache_fields,
        define_constructor=False)

@njit(cache=True)
def ContinuousSplitCache_ctor(n_classes):
    st = new(ContinuousSplitCacheType)
    st.best_thresh = f4(0.0)
    st.y_counts_l = np.zeros(n_classes, dtype=np.uint32)
    st.y_counts_r = np.zeros(n_classes, dtype=np.uint32)
    return st

# Keep old name as alias for backward compatibility
ContinousSplitCache = ContinuousSplitCache
ContinousSplitCacheType = ContinuousSplitCacheType
