from numbaILP.structref import define_structref, define_structref_template
from numba import njit
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type

''' DataStats stores the inputted features and labels re-encoded so that the
    the nominal values are remapped to be contiguous (i.e. 0,1,2,3,... etc.).
    Structures for performing this remapping are also stored here. Additionally 
    sorted copies of the continous features are stored here. Lastly 
    data stats contains summary statistics.  
''' 


#### DataStats ####

# SortData is a mutable struct to keep sorted features and indicies 
SortData, SortDataType =  \
    define_structref("SortData", [('inds', i4[::1]), ('vals',f4[::1])]) 

data_stats_fields = [
    
    ### The Contiguously Encoded Data ###

    # The feature matrix for nominal values, but remapped 
    #  to contiguous values if necessary
    ('X_nom', i4[:,:]),

    # The feature matrix for continuous values
    ('X_cont', f4[:,:]),

    # The label for each sample, remapped to contiguous values 
    #  if necessary
    ('Y', i4[:]),

    ### Remapping (to make contiguous) / Reordering structures ###

    # Maps y labels to [0,...]
    ('y_map', DictType(i4,i4)),
    # Maps each nominal feature to [0,...]
    ('nom_v_maps', ListType(DictType(i4,i4))),

    # Maps encoded y labels back to original form
    ('y_inv_map', DictType(i4,i4)),
    # Maps encoded nominal features back to original form
    ('nom_v_inv_maps', ListType(DictType(i4,i4))),

    # Sorted indicies and values for each continous feature
    ('X_cont_ft_sort_data', ListType(SortDataType)),

    ### Summary Stats ###

    # The counts of each class
    ('y_counts', u4[::1]),

    # The number of unique values per nominal feature 
    ('n_vals', i4[::1]),

    # The total number of samples in the data
    ('n_samples', i4),

    # The total number of classes in the data
    ('n_classes', i4),

    # The total number of nominal features
    ('n_nom_features', i4),

    # The total number of continous features
    ('n_cont_features', i4),

    # If true then this has been initialized 
    ('is_initialized', u1),

    ### Control flags to reduce redundant processing ###

    # When user sets to true they promise incoming y labels start at 
    #  zero and are contiguous. 
    ('y_contiguous', u1),

    # When user sets to true they promise all incoming nominal values 
    #  start at zero and are contiguous.
    ('nom_v_contiguous', u1),
]

DataStats, DataStatsType = \
    define_structref("DataStats",data_stats_fields,
         define_constructor=False) 

i4_i4_dict = DictType(i4,i4)

@njit(cache=True)
def DataStats_ctor(y_contiguous=False,nom_v_contiguous=False):
    ''' Constructor for an empty data stats object ''' 
    st = new(DataStatsType)

    st.y_map = Dict.empty(i4,i4)
    st.y_inv_map = Dict.empty(i4,i4)
    st.nom_v_maps = List.empty_list(i4_i4_dict)
    st.nom_v_inv_maps = List.empty_list(i4_i4_dict)
    st.X_cont_ft_sort_data = List.empty_list(SortDataType)
    st.y_counts = np.zeros(1,dtype=np.uint32)
    st.is_initialized = False
    st.y_contiguous = y_contiguous
    st.nom_v_contiguous = nom_v_contiguous

 
    return st

@njit(cache=True)
def insert_ds_dicts(ds, x_nom, x_cont, y):
    ''' Insert a sample into the nominal value and y label mapping dictionaries'''
    if(not ds.y_contiguous and y not in ds.y_map):
        l = len(ds.y_map)
        ds.y_map[y] = l
        ds.y_inv_map[l] = y

        new_y_counts = np.empty(l,dtype=np.uint32)
        new_y_counts[:l-1] = ds.y_counts
        new_y_counts[l] = 1
        ds.y_counts = new_y_counts
    else:
        ds.y_counts[ds.y_map[y]] += 1

    if(not ds.nom_v_contiguous):
        for j, x_n in enumerate(x_nom):
            mp, inv_mp = ds.nom_v_maps[j], ds.nom_v_inv_maps[j]
            if(x_n not in mp):
                l = len(mp)
                inv_mp[l] = x_n
                mp[x_n] = l


@njit(cache=True)
def update_summary_stats(ds):
    ''' Update summary helper stats '''
    if(not ds.nom_v_contiguous):
        ds.n_vals = np.empty(len(ds.nom_v_maps), dtype=np.int32)
        for j, nv_map in enumerate(ds.nom_v_maps):
            ds.n_vals[j] = len(nv_map)
    else:
        ds.n_vals = np.empty(ds.X_nom.shape[1], dtype=np.int32)
        for j in range(ds.X_nom.shape[1]):
            ds.n_vals[j] = max(ds.X_nom[:,j])+1

            
    ds.n_samples = len(ds.Y)
    ds.n_nom_features = ds.X_nom.shape[1]
    ds.n_cont_features = ds.X_cont.shape[1]



@njit(cache=True)
def init_data_stats(ds, X_nom, X_cont, Y):
    ''' Initialize the data stats for a fit() '''
    for j in range(X_nom.shape[1]):
        ds.nom_v_maps.append(Dict.empty(i4,i4))
        ds.nom_v_inv_maps.append(Dict.empty(i4,i4))

    for i in range(len(X_nom)):
        insert_ds_dicts(ds, X_nom[i], X_cont[i], Y[i])


@njit(cache=True)
def update_data_stats(ds, x_nom, x_cont, y):
    ''' Update the data stats for an ifit() '''
    for i in range(0, len(x_nom)-len(ds.nom_v_maps))
        ds.nom_v_maps.append(Dict.empty(i4,i4))
    for i in range(0, len(x_nom)-len(ds.nom_v_maps))
        ds.nom_v_maps.append(Dict.empty(i4,i4))

    insert_ds_dicts(ds, x_nom, x_cont, y)
