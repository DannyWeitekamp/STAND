from numbaILP.structref import define_structref, define_structref_template
from numba import njit
from numba import optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type
from numba.experimental.structref import new
import numpy as np

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
    ('X_nom', i4[:,::1]),

    # The feature matrix for continuous values
    ('X_cont', f4[:,::1]),

    # The label for each sample, remapped to contiguous values 
    #  if necessary
    ('Y', i4[::1]),

    # When ifit_enabled keep around buffers to avoid excessive copying
    ('X_nom_buffer', i4[:,::1]),
    ('X_cont_buffer', f4[:,::1]),
    ('Y_buffer', i4[::1]),

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

    # A list of unique classes
    ('u_ys', i4[::1]),

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

    # If true then the parent tree has ifit_enabled
    ('ifit_enabled', u1),
]

DataStats, DataStatsType = \
    define_structref("DataStats",data_stats_fields,
         define_constructor=False) 

i4_i4_dict = DictType(i4,i4)

@njit(cache=True)
def DataStats_ctor(nom_v_contiguous=False, y_contiguous=False, ifit_enabled=False):
    ''' Constructor for an empty data stats object ''' 
    st = new(DataStatsType)

    st.y_map = Dict.empty(i4,i4)
    st.y_inv_map = Dict.empty(i4,i4)
    st.nom_v_maps = List.empty_list(i4_i4_dict)
    st.nom_v_inv_maps = List.empty_list(i4_i4_dict)
    st.X_cont_ft_sort_data = List.empty_list(SortDataType)
    st.u_ys = np.empty(0, dtype=np.int32)
    st.y_counts = np.zeros(1,dtype=np.uint32)
    st.is_initialized = False
    st.y_contiguous = y_contiguous
    st.nom_v_contiguous = nom_v_contiguous
    st.ifit_enabled = ifit_enabled

    st.X_nom = st.X_nom_buffer = np.empty((0,0),dtype=np.int32)
    st.X_cont = st.X_cont_buffer = np.empty((0,0),dtype=np.float32)
    st.Y = st.Y_buffer = np.empty(0,dtype=np.int32)
    st.n_vals = np.empty(0,dtype=np.int32)

    st.n_classes = 0

    return st

@njit(cache=True)
def _insert_w_ds_maps(i, ds, x_nom, y):
    ''' Insert a sample into the nominal value and y label mapping dictionaries'''
    # print(ds.y_counts, y, ds.y_contiguous)
    if(ds.y_contiguous):
        ds.y_counts[y] += 1
    else:
        if(y not in ds.y_map):
            l = i4(len(ds.y_map))
            ds.y_map[y] = l
            ds.y_inv_map[l] = y

            new_y_counts = np.empty(l+1,dtype=np.uint32)
            new_y_counts[:l] = ds.y_counts
            new_y_counts[l] = 0
            # print("B",ds.y_counts, new_y_counts)
            ds.y_counts = new_y_counts

        y_mapped = ds.y_map[y]
        # print(ds.y_counts, y_mapped)
        ds.y_counts[y_mapped] += 1
        ds.Y[i] = y_mapped
        # print('p', len(ds.Y), i)

    if(not ds.nom_v_contiguous):
        for j, x_n in enumerate(x_nom):
            # print(i,j,len(ds.nom_v_maps))
            mp, inv_mp = ds.nom_v_maps[j], ds.nom_v_inv_maps[j]
            if(x_n not in mp):
                l = len(mp)
                inv_mp[l] = x_n
                mp[x_n] = l

            ds.X_nom[i,j] = mp[x_n]

    # print("DONE")

    return 0


@njit(cache=True)
def _update_summary_stats_reinit(ds):
    ''' Update summary helper stats '''
    if(not ds.nom_v_contiguous):
        ds.n_vals = np.empty(len(ds.nom_v_maps), dtype=np.int32)
        for j, nv_map in enumerate(ds.nom_v_maps):
            ds.n_vals[j] = len(nv_map)
    else:
        ds.n_vals = np.empty(ds.X_nom.shape[1], dtype=np.int32)
        for j in range(ds.X_nom.shape[1]):
            mx = max(ds.X_nom[:,j]) if len(ds.X_nom[:,j]) > 0 else 0
            ds.n_vals[j] = mx+1

            
    ds.n_samples = len(ds.Y)
    ds.n_nom_features = ds.X_nom.shape[1]
    ds.n_cont_features = ds.X_cont.shape[1]

    ds.n_classes = len(ds.y_counts)

    if(ds.y_contiguous):
        ds.u_ys = np.arange(ds.n_classes, dtype=np.int32)
    else:
        ds.u_ys = np.empty(ds.n_classes, dtype=np.int32)
        for i, v in enumerate(ds.y_map.keys()):
            ds.u_ys[i] = v


@njit(cache=True)
def _assign_buffers(ds):
    ds.X_nom_buffer = ds.X_nom
    ds.X_cont_buffer = ds.X_cont
    ds.Y_buffer = ds.Y


@njit(cache=True)
def reinit_datastats(ds, X_nom, X_cont, Y):
    ''' Initialize the data stats for a fit()  '''

    # print(X_nom, X_cont, Y)
    for j in range(X_nom.shape[1]):
        ds.nom_v_maps.append(Dict.empty(i4,i4))
        ds.nom_v_inv_maps.append(Dict.empty(i4,i4))

    if(ds.nom_v_contiguous):
        # Not sure why but need to copy to avoid error
        ds.X_nom = X_nom.astype(np.int32).copy()
    else:
        ds.X_nom = np.empty(X_nom.shape,dtype=np.int32)

    ds.X_cont = X_cont

    if(ds.y_contiguous):
        ds.Y = Y
        ds.y_counts = np.zeros(max(Y)+1,dtype=np.uint32)
    else:
        ds.Y = np.empty(Y.shape,dtype=np.int32)
        ds.y_counts = np.zeros(0,dtype=np.uint32)

    for i in range(len(X_nom)):
        _insert_w_ds_maps(i, ds, X_nom[i], Y[i])

    _update_summary_stats_reinit(ds)
    if(ds.ifit_enabled): _assign_buffers(ds)
    ds.is_initialized = True
    return 0

MAX_BUFFER_EXPAND = 1024
START_BUFFER_SIZE = 8

@njit(cache=True)
def _expand_buffers(ds, x_nom, x_cont, y):
    if(len(x_nom) > len(ds.n_vals)):
        new_n_vals = np.empty(len(x_nom),dtype=np.int32)
        new_n_vals[:len(ds.n_vals)] = ds.n_vals
        new_n_vals[len(ds.n_vals):] = 0
        # print(ds.n_vals, '->', new_n_vals)
        ds.n_vals = new_n_vals


    l,w = ds.X_nom.shape
    if(len(ds.X_nom_buffer) <= l or
        len(x_nom) > ds.X_nom_buffer.shape[1]):
        new_w = max(len(x_nom), ds.X_nom_buffer.shape[1])
        new_l = min(len(ds.X_nom_buffer)*2,len(ds.X_nom_buffer)+MAX_BUFFER_EXPAND)
        new_l = max(new_l, START_BUFFER_SIZE)
        ds.X_nom_buffer = np.empty((new_l ,new_w), dtype=np.int32)
        ds.X_nom_buffer[:l,:w] = ds.X_nom
        ds.X_nom_buffer[:l,w:] = 0
    ds.X_nom = ds.X_nom_buffer[:l+1]
    # print((l,w), "-==>", ds.X_nom.shape)

    l,w = ds.X_cont.shape
    if(len(ds.X_cont_buffer) <= l or
        len(x_cont) > ds.X_cont_buffer.shape[1]):

        new_w = max(len(x_cont), ds.X_cont_buffer.shape[1])
        new_l = min(len(ds.X_cont_buffer)*2,len(ds.X_cont_buffer)+MAX_BUFFER_EXPAND)
        new_l = max(new_l, START_BUFFER_SIZE)
        ds.X_cont_buffer = np.empty((new_l ,new_w), dtype=np.float32)
        ds.X_cont_buffer[:l,:w] = ds.X_cont
        ds.X_cont_buffer[:l,w:] = 0
    # print("MEEP", ds.X_cont_buffer.shape,l+1)
    ds.X_cont = ds.X_cont_buffer[:l+1]
    # print("SHEMEEP", ds.X_cont.shape)

    l = len(ds.Y)
    if(len(ds.Y_buffer) <= l):
        new_l = min(len(ds.Y_buffer)*2,len(ds.Y_buffer)+MAX_BUFFER_EXPAND)
        new_l = max(new_l, START_BUFFER_SIZE)
        ds.Y_buffer = np.empty(new_l, dtype=np.int32)
        ds.Y_buffer[:l] = ds.Y
        ds.Y = ds.Y_buffer[:l]
    ds.Y = ds.Y_buffer[:l+1]
    # print("FLEEP", ds.Y.shape)



@njit(cache=True)
def _update_summary_stats_update(ds):
    ''' Update summary helper stats '''

    if(not ds.nom_v_contiguous):
        # ds.n_vals = np.empty(len(ds.nom_v_maps), dtype=np.int32)
        for j, nv_map in enumerate(ds.nom_v_maps):
            ds.n_vals[j] = len(nv_map)
    else:
        # print(ds.X_nom)
        x_nom = ds.X_nom[-1]
        # print("x_nom", x_nom)
        for j in range(ds.X_nom.shape[1]):
            if(x_nom[j]+1 > ds.n_vals[j]):
                ds.n_vals[j] = x_nom[j]+1 
            
    ds.n_samples = len(ds.Y)
    ds.n_nom_features = ds.X_nom.shape[1]
    ds.n_cont_features = ds.X_cont.shape[1]
    prev_n_classes = ds.n_classes
    ds.n_classes = len(ds.y_counts)

    if(prev_n_classes != ds.n_classes):
        if(ds.y_contiguous):
            ds.u_ys = np.arange(ds.n_classes, dtype=np.int32)
        else:
            ds.u_ys = np.empty(ds.n_classes, dtype=np.int32)
            for i, v in enumerate(ds.y_map.keys()):
                ds.u_ys[i] = v

@njit(cache=True)
def update_data_stats(ds, x_nom, x_cont, y):
    ''' Update the data stats for an ifit() '''
    # if(not ds.is_initialized):
    n_samples = len(ds.Y)
    for i in range(0, len(x_nom)-len(ds.nom_v_maps)):
        ds.nom_v_maps.append(Dict.empty(i4,i4))
        ds.nom_v_inv_maps.append(Dict.empty(i4,i4))
    for i in range(0, len(x_nom)-len(ds.nom_v_maps)):
        ds.nom_v_maps.append(Dict.empty(i4,i4))
        ds.nom_v_inv_maps.append(Dict.empty(i4,i4))

    # print("A",ds.y_contiguous)

    if(ds.y_contiguous):
        if(y >= len(ds.y_counts)):
            y_counts_new = np.empty(y+1,dtype=np.uint32)
            y_counts_new[:y] = ds.y_counts
            y_counts_new[y] = 0
            ds.y_counts = y_counts_new

    # print("B")

    _expand_buffers(ds, x_nom, x_cont, y)
    _insert_w_ds_maps(n_samples, ds, x_nom, y)
    # print("C")
    if(ds.nom_v_contiguous):
        ds.X_nom[n_samples] = x_nom
        ds.Y[n_samples] = y
    ds.X_cont[n_samples] = x_cont
    

    # print("D")
    # print(ds.n_vals)
    _update_summary_stats_update(ds)
    ds.is_initialized = True
    return 0
