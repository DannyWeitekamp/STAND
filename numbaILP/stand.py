import numpy as np
from numbaILP.structref import define_structref, define_structref_template
from numbaILP.splitter import TreeClassifier, str_tree, fit_tree, _count_branches, _count_covering_branches, decode_split, encode_split, filter_leaves
from numbaILP.tree_structs import TreeNodeType
from numba import config, njit, threading_layer, types
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType,ListType, unicode_type, NamedTuple,NamedUniTuple,Tuple,literal
from numba.experimental.structref import new


# specific_ext_fields = [
#     ('leaf_index', i8),
#     ('ext_size', u8),
#     ('enc_splits', u8[::1]),
# ]

# SpecificExt, SpecificExtType = define_structref("SpecificExt", specific_ext_fields, define_constructor=True)


# np_specific_ext_type = np.dtype([
#     ('is_cont', np.uint8),
#     ('negated', np.uint8),
#     ('split', np.int32),
#     ('val', np.int32),
#     ('ext_size', np.uint16)
# ])

# specific_ext_type = numba.from_dtype(np_specific_ext_type)
# print(specific_ext_type)

u8_arr_u8_tup_typ = Tuple((u8[::1],u8))

stand_fields = [
    # The option tree that characterizes the general set G
    ('op_tree', types.Any),

    # The specifc extensions of G that characterizes S. Maps the node index 
    #  of each positive leaf (i.e. leaves that captures at least one positive 
    #  training instance) to a tuple (spec_ext, ext_size). The first argument
    #  "spec_ext" is the set of conditions that select the invariant 
    #  features of the instances selected by that leaf (i.e. it's terminal subset).
    #  These conditions are encoded as 64 unisigned ints using encode_split().
    #  "ext_size" is the how many of the conditions in the specific extension,
    #  are not also in the parents branches of the positive leaf. 
    ('spec_exts', DictType(i4,Tuple((u8[::1],u8)))),

    # The positive class 
    ('positive_class', i4),
]

STAND, STANDTypeTemplate = define_structref_template("STAND", stand_fields, define_constructor=False)


class STANDClassifier(object):
    def __init__(self, positive_class=1, **kwargs):
        self.op_tree_classifier = TreeClassifier(preset_type='option_tree', **kwargs)
        self.op_tree = self.op_tree_classifier.tree
        self.stand_type = self.gen_stand_type(self.op_tree_classifier.tree_type)
        self.stand = STAND_ctor(self.stand_type, self.op_tree, positive_class)

    def gen_stand_type(self, tree_type):
        sf = [('op_tree', tree_type), *stand_fields[1:]]
        return STANDTypeTemplate(sf)

    def fit(self, X_nom, X_cont, Y, miss_mask=None, ft_weights=None):
        self.op_tree_classifier.fit(X_nom, X_cont, Y, miss_mask, ft_weights)
        fit_spec_ext(self.stand)

    def ifit(self, x_nom, x_cont, y, miss_mask=None, ft_weights=None):
        if(x_nom is None): x_nom = np.empty((0,), dtype=np.int32)
        if(x_cont is None): x_cont = np.empty((0,), dtype=np.float32)
        self.op_tree_classifier.fit(x_nom, x_cont, y, miss_mask, ft_weights)
        fit_spec_ext(self.stand)

    def instance_ambiguity(self, x_nom=None, x_cont=None):
        if(x_nom is None): x_nom = np.empty((0,), dtype=np.int32)
        if(x_cont is None): x_cont = np.empty((0,), dtype=np.float32)
        return instance_ambiguity(self.stand, x_nom, x_cont)

    def __str__(self):
        return str_tree(self.op_tree)



u8_arr = u8[::1]

@njit(cache=True)
def STAND_ctor(stand_type, op_tree, positive_class):
    st = new(stand_type)
    st.op_tree = op_tree
    st.positive_class = positive_class
    st.spec_exts = Dict.empty(i4, u8_arr_u8_tup_typ)
    return st

@njit(cache=True)
def calc_invariant_nom_mask(X_nom):
    nom_invariants = np.ones(X_nom.shape[1],dtype=np.uint8)
    x0 = X_nom[0]
    for i in range(1,len(X_nom)):
        X_nom_i = X_nom[i]
        for j in range(len(x0)):
            if(nom_invariants[j]):
                nom_invariants[j] &= (x0[j] == X_nom_i[j])


    return nom_invariants


@njit(cache=True)
def fit_spec_ext(stand):
    ''' 
    Builds specific extension for each positive leaf of the fitted option tree. 
    A positive leaf is a leaf that contains some positive instances.
    '''
    tree = stand.op_tree
    stand.spec_exts = Dict.empty(i4, u8_arr_u8_tup_typ)

    # TODO: Check edge case when the training set doesn't contain the positive class
    pc = tree.data_stats.y_map[stand.positive_class]
    X_nom = tree.data_stats.X_nom

    for leaf in tree.leaves:    
        if(leaf.counts[pc] > 0):
            trm_ss_nom = X_nom[leaf.sample_inds]
            x_nom_0 = trm_ss_nom[0]
            nom_invt_mask = calc_invariant_nom_mask(trm_ss_nom)
            L = ext_size = np.sum(nom_invt_mask, dtype=np.int64)
            spec_ext = np.empty(L, dtype=np.uint64)

            # "ext_size" is the number of conditions in the specific extension
            #   that are not present in any branch of "leaf". Decrement any 
            #   repetitions found in the these branches.
            n_branches = _count_branches(tree, leaf)
            for node_ind, n in enumerate(n_branches):
                if(n > 0):
                    node = tree.nodes[node_ind]
                    for sd in node.split_data:
                        split, val = sd.split_ind, sd.val
                        if(nom_invt_mask[split] and x_nom_0[split]==val):
                            ext_size -= 1

            # Build "spec_ext" the conditions for the specific extention of "leaf". 
            c = 0
            for split, is_invariant in enumerate(nom_invt_mask):
                if(is_invariant):
                    val = x_nom_0[split]
                    spec_ext[c] = encode_split(0,0,i4(split),val) 
                    c += 1

            # Insert extension and size into "spec_exts" dict of the STAND structref
            assert ext_size > 0 
            stand.spec_exts[leaf.index] = (spec_ext, u8(ext_size))


@njit(cache=True)
def instance_ambiguity(stand, x_nom, x_cont):    
    tree = stand.op_tree
    pc = tree.data_stats.y_map[stand.positive_class]
    nom_v_maps = tree.data_stats.nom_v_maps
    
    leaves = filter_leaves(tree, x_nom, x_cont)
    print("NLEAVES:", len(leaves))

    A_px = 0
    A_nx = 0
    Nn, Np = 0, 0
    for leaf in leaves:
        n_branches = _count_covering_branches(tree, leaf, x_nom, x_cont)
        # The number of parent branches leading into that 
        #  don't (_nn) and do (_np) select (x_nom, x_cont)
        nn_np = n_branches[leaf.index]
        _nn, _np = nn_np[0], nn_np[1]
        print("::", _nn, _np)
        Nn += _nn
        Np += _np

        # Positive leaf case 
        if(leaf.counts[pc] > 0):
            # Find the number of conditions failed in the specific extension 
            n_ext_failed = 0
            spec_ext, ext_size = stand.spec_exts[leaf.index]
            for enc_split in spec_ext:
                is_cont, negated, split, val = decode_split(enc_split)
                if(not is_cont):
                    mapped_val = nom_v_maps[split][x_nom[split]]
                    if(mapped_val != val): n_ext_failed 
                else:
                    pass

            A_px += _nn * (1 + ext_size)
            A_px += _np * (n_ext_failed)

            # print(_nn, ext_size, _np, n_ext_failed)
            # print(n_branches)

        # Negative Leaf case 
        else:
            A_nx += _np * (1+ext_size)
    
    print(":", A_px, A_nx, Np, Nn)
    
    den = (Np + Nn)
    if(den == 0): return 0.0

    p = Np / den
    # print(p, A_px, A_nx)
    return p*A_px + (1-p)*A_nx


        
        









