from numbaILP.structref import define_structref, define_structref_template
from numba import njit
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type


''' A split cache is responsible for storing information about candidate splits
    at particular nodes in the tree. If ifit_enabled=True then these are stored
    in the context_cache of the the tree. 
''' 

#### NominalSplitCache ####

nominal_split_cache_fields = [
    ('best_v', i4),
    ('v_counts', u4[:]),
    ('y_counts_per_v', u4[:,:]),
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



#### ContinousSplitCache ####

continous_split_cache_field = [
    ('is_const', u1),
    ('threshold', f8),
    ('op', i4),
    ('left_counts', i4[:]),
    ('right_counts', i4[:]),
    ('nan_counts', i4[:]),
]

ContinousSplitCache, ContinousSplitCacheType = \
    define_structref("ContinousSplitCache", continous_split_cache_field,
        define_constructor=False) 
