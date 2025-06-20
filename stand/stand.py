import numpy as np
from stand.structref import define_structref, define_structref_template
from stand.tree_classifier import (
    TreeClassifier, str_tree, fit_tree, _count_branches,
    _count_covering_branches, decode_split, encode_split,
    filter_leaves, get_branch_splits, prob_item_type)
from stand.tree_structs import TreeNodeType
from stand.split_caches import NominalSplitCacheType
from stand.utils import _struct_from_pointer
from numba import config, njit, threading_layer, types
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType,ListType, unicode_type, NamedTuple,NamedUniTuple,Tuple,literal
from numba.experimental.structref import new

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')

# from numba import njit, objmode
# import ctypes

# CLOCK_MONOTONIC = 0x1
# clock_gettime_proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int,
#                                        ctypes.POINTER(ctypes.c_long))
# pybind = ctypes.CDLL(None)
# clock_gettime_addr = pybind.clock_gettime
# clock_gettime_fn_ptr = clock_gettime_proto(clock_gettime_addr)


# @njit
# def timenow():
#     timespec = np.zeros(2, dtype=np.int64)
#     clock_gettime_fn_ptr(CLOCK_MONOTONIC, timespec.ctypes)
#     ts = timespec[0]
#     tns = timespec[1]
#     return np.float64(ts) + 1e-9 * np.float64(tns)




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

spec_ext_data_t = Tuple((u8[::1],f4[::1], u4, f4))

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
    ('spec_exts', DictType(i4, spec_ext_data_t)),

    # The positive class 
    ('positive_class', i4),
]

STAND, STANDTypeTemplate = define_structref_template("STAND", stand_fields, define_constructor=False)


class STANDClassifier(object):
    def __init__(self, positive_class=1, pred_kind="max_leaves", **kwargs):
        kwargs['split_choice'] = kwargs.get('split_choice', 'dyn_all_near_max')
        # print("SPLIT CHOICE:", kwargs['split_choice'])
        self.pred_kind = pred_kind
        self.op_tree_classifier = TreeClassifier(preset_type='option_tree', **kwargs)
        self.op_tree = self.op_tree_classifier.tree
        self.stand_type = self.gen_stand_type(self.op_tree_classifier.tree_type)
        self.stand = STAND_ctor(self.stand_type, self.op_tree, positive_class)


    def gen_stand_type(self, tree_type):
        sf = [('op_tree', tree_type), *stand_fields[1:]]
        return STANDTypeTemplate(sf)

    def fit(self, X_nom, X_cont, Y, miss_mask=None, ft_weights=None):
        # with PrintElapse("fit option_tree"):
        self.op_tree_classifier.fit(X_nom, X_cont, Y, miss_mask, ft_weights)
        # with PrintElapse("fit_spec_ext"):
        try:
            fit_spec_ext(self.stand)
        except Exception as e:
            print(self)
            raise e
        # print("N NODES:", len(self.op_tree_classifier.nodes))

    # TODO : ADD SPECIFIC CHECK
    def predict(self, X_nom, X_cont):
        if(self.stand is None): raise RuntimeError("STANDClassifier must be fit before predict_prob() is called.")
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)

        if(self.pred_kind == "prob" or self.pred_kind == "density"):
            y_density, probs = stand_predict_y_density(self.stand, X_nom, X_cont)
            probs, labels = stand_predict_prob(self.stand, X_nom, X_cont)
            # print(probs)
            # print(probs.shape)
            if(self.pred_kind == "density"):
                return labels[np.argmax(y_density, axis=-1)] #self.op_tree_classifier.predict(X_nom, X_cont)
            else:
                return labels[np.argmax(probs, axis=-1)] #self.op_tree_classifier.predict(X_nom, X_cont)
        else:
            return self.op_tree_classifier.predict(X_nom, X_cont)

    def predict_prob(self, X_nom, X_cont):
        if(self.stand is None): raise RuntimeError("STANDClassifier must be fit before predict_prob() is called.")
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        return stand_predict_prob(self.stand, X_nom, X_cont)
        # self.op_tree_classifier.predict_prob(X_nom, X_cont)

    def instance_certainty(self, X_nom, X_cont):
        if(self.stand is None): raise RuntimeError("STANDClassifier must be fit before predict_prob() is called.")
        if(X_nom is None): X_nom = np.empty((0,0), dtype=np.int32)
        if(X_cont is None): X_cont = np.empty((0,0), dtype=np.float32)
        X_nom = X_nom.astype(np.int32)
        X_cont = X_cont.astype(np.float32)
        return instance_certainty(self.stand, X_nom, X_cont)

    def ifit(self, x_nom, x_cont, y, miss_mask=None, ft_weights=None):
        if(x_nom is None): x_nom = np.empty((0,), dtype=np.int32)
        if(x_cont is None): x_cont = np.empty((0,), dtype=np.float32)
        self.op_tree_classifier.fit(x_nom, x_cont, y, miss_mask, ft_weights)
        
        fit_spec_ext(self.stand)

    def instance_ambiguity(self, x_nom=None, x_cont=None):
        if(x_nom is None): x_nom = np.empty((0,), dtype=np.int32)
        if(x_cont is None): x_cont = np.empty((0,), dtype=np.float32)
        return instance_ambiguity(self.stand, x_nom, x_cont)

    def __str__(self, **kwargs):
        return self.op_tree_classifier.__str__(**kwargs)

    def get_lit_priorities(self):
        return self.op_tree_classifier.get_lit_priorities()

    def get_opt_conjs_for_label(self, label):
        return self.op_tree_classifier.get_opt_conjs_for_label(label)

    def get_conds(self, label, literals="all", conjuncts="all"):
        out = self.op_tree_classifier.get_conds(
                label, literals, conjuncts)        
        return out


u8_arr = u8[::1]


@njit(cache=True)
def STAND_ctor(stand_type, op_tree, positive_class):
    st = new(stand_type)
    st.op_tree = op_tree
    st.positive_class = positive_class
    st.spec_exts = Dict.empty(i4, spec_ext_data_t)
    return st

@njit(cache=True)
def calc_invariant_nom_mask(X_nom):
    nom_invariants = np.ones(X_nom.shape[1],dtype=np.uint8)
    x0 = X_nom[0]
    for i in range(1,len(X_nom)):
        nom_invariants &= (x0 == X_nom[i])
    return nom_invariants

@njit(cache=True)
def calc_spec_ext_weight(tree, leaf, enc_split):
    is_cont, negated, split, val = decode_split(enc_split)

    lam = tree.lam_e
    n_samples = len(leaf.sample_inds)
    avg_par_v_prob = 0.0 #np.zeros(v_counts.shape, dtype=np.float32)


    # if(n_samples == 0):
    #     print(str_tree(tree))

    self_w = 1.0 / (1.0 + lam / n_samples)

    # print()
    y = np.argmax(leaf.counts)

    if(len(leaf.parents) > 0):
        for i, (p_node_ind, enc_split) in enumerate(leaf.parents):
            p_node = tree.nodes[p_node_ind]

            if(p_node_ind == 0):
                p_w = 1.0
            else:
                p_w = 1.0 / (1.0 + lam/len(p_node.sample_inds))

            par_cache_ptr = p_node.nominal_split_cache_ptrs[split]
            par_spl_c = _struct_from_pointer(NominalSplitCacheType, par_cache_ptr)
            
            # par_tot = np.sum(par_spl_c.par_w_v_probs)

            # print("A")
            # --- OLD ----
            # avg_par_v_prob += par_spl_c.par_w_v_probs[val] #/ par_tot if par_tot != 0.0 else 0.0            
            # avg_par_v_prob += (p_w-self_w) * par_spl_c.w_v_probs[val] #/ np.sum(par_spl_c.w_v_counts))
            # 
            #######

            # Note: This has a slight benefit to productive monotonicity over "w_v_probs" 
            avg_par_v_prob += par_spl_c.par_w_y_probs_per_v[val, y] #/ par_tot if par_tot != 0.0 else 0.0            
            avg_par_v_prob += (p_w-self_w) * par_spl_c.w_y_probs_per_v[val, y] #/ np.sum(par_spl_c.w_v_counts))

            ######

            # if(split == 7):
            #     print("-----", split, "==", val, ":", n_samples, len(p_node.sample_inds), self_w, p_w, (p_w-self_w))
            #     print(f"NODE={leaf.index}", f"PAR={p_node_ind}", ":", par_spl_c.par_w_v_probs[val])
            #     print("B", p_w, self_w, par_spl_c.w_v_probs[val], np.sum(par_spl_c.w_v_probs))
            #     print("avg_par_v_prob @ 7:", avg_par_v_prob)

            # print("C")
        
        # print("D")
        avg_par_v_prob /= len(leaf.parents)
        # print("avg_par_v_prob:", avg_par_v_prob, self_w)
        
        

        w_v_prob = (avg_par_v_prob + self_w)
        # if(split == 7):
        #     print("w_v_prob:", w_v_prob)
        # if(split == 7):
        #     print("avg_par_v_prob @ 7:", w_v_prob)
        # if(split == 2):
        #     print("avg_par_v_prob @ 2:", w_v_prob)

    else:
        w_v_prob = 1.0



    return (n_samples + .5 + w_v_prob) / (n_samples + 2)



@njit(cache=True)
def fit_spec_ext(stand):
    ''' 
    Builds specific extension for each positive leaf of the fitted option tree. 
    A positive leaf is a leaf that contains some positive instances.
    '''
    tree = stand.op_tree
    stand.spec_exts = Dict.empty(i4, spec_ext_data_t)

    # TODO: Check edge case when the training set doesn't contain the positive class
    # pc = tree.data_stats.y_map[stand.positive_class]
    X_nom = tree.data_stats.X_nom

    for leaf in tree.leaves:
        # if(leaf.counts[pc] > 0):
        trm_ss_nom = X_nom[leaf.sample_inds]
        x_nom_0 = trm_ss_nom[0]

        nom_invt_mask = calc_invariant_nom_mask(trm_ss_nom)

        L = np.sum(nom_invt_mask, dtype=np.int64)
        spec_ext = np.empty(L, dtype=np.uint64)
        spec_ws = np.empty(L, dtype=np.float32)

        branch_splits = get_branch_splits(tree, leaf)
        # Build "spec_ext" the conditions for the specific extention of "leaf". 
        # "ext_size" is the number of conditions in the specific extension
        #   that are not present in any branch of "leaf". Decrement any 
        #   repetitions found in the these branches.
        c = 0
        total_w = f4(0.0)
        ext_size = u4(0)
        for split, is_invariant in enumerate(nom_invt_mask):
            if(is_invariant):
                val = x_nom_0[split]
                spec_ext[c] = enc_split = encode_split(0,0,i4(split),val) 
                
                ext_w = calc_spec_ext_weight(tree, leaf, enc_split)
                spec_ws[c] = ext_w
                c += 1

                if(enc_split not in branch_splits):
                    ext_size += 1
                    total_w += ext_w

        # for split_enc in branch_splits:
        #     is_cont, negated, split, val = decode_split(split_enc)
        #     if(nom_invt_mask[split] and negated ^ (x_nom_0[split]==val)):
        #         ext_size -= 1

        # Insert extension and size into "spec_exts" dict of the STAND structref
        assert ext_size >= 0 and ext_size <= L
        stand.spec_exts[leaf.index] = (spec_ext, spec_ws, u4(ext_size), f4(total_w))

@njit(cache=True)
def eval_specific_extension(stand, leaf, x_nom, x_cont):
    nom_v_maps = stand.op_tree.data_stats.nom_v_maps
    
    if(leaf.index not in stand.spec_exts):
        return 0, 0, 0, 0.0, 0.0

    spec_ext, ext_ws, ext_size, ext_weight = stand.spec_exts[leaf.index]
    n_ext_matches = 0
    n_ext_fails = 0
    w_ext_matches = 0.0
    w_ext_fails = 0.0
    for enc_split, ext_w in zip(spec_ext, ext_ws):
        is_cont, negated, split, val = decode_split(enc_split)
        if(is_cont):
            # Not implemented
            pass
        else:
            mapped_val = nom_v_maps[split].get(x_nom[split],-1)
            if(mapped_val == val):
                n_ext_matches += 1
                w_ext_matches += ext_w
                w_ext_fails += 1.0-ext_w
            else:
                n_ext_fails += 1
                w_ext_matches += 1.0-ext_w
                w_ext_fails += ext_w
    return ext_size, n_ext_matches, n_ext_fails, w_ext_matches, w_ext_fails

@njit(cache=True)
def stand_predict_y_density(stand, X_nom, X_cont):
    # NOTE: Should I really call this a probability? It's not a normalized one.
    tree = stand.op_tree
    L = max(len(X_nom),len(X_cont))
    if(len(X_nom) == 0): X_nom = np.empty((L,0), dtype=np.int32)
    if(len(X_cont) == 0): X_cont = np.empty((L,0), dtype=np.float32)
    
    y_uvs = tree.data_stats.u_ys

    lam = tree.lam_l

    # out = np.zeros((L,len(y_uvs)),dtype=prob_item_type)
    out_probs = np.zeros((L,len(y_uvs)),dtype=np.float64)
    probs = np.zeros((L,len(y_uvs)),dtype=np.float64)
    y_density = np.zeros((L,len(y_uvs)),dtype=np.float64)
    tot_leaf_weight = np.zeros((L,len(y_uvs)),dtype=np.float64)
    zz_max = np.zeros((L,len(y_uvs)),dtype=np.float64)
    # For each sample i, filter it into leaves and compute
    #  the probability of correctness on the basis of the specific extension 

    
    w_100 = 1/(1.0+lam/100)
    w_root = 1/(1.0+lam/ len(tree.nodes[0].sample_inds))

    for i in range(L):
        x_nom, x_cont = X_nom[i], X_cont[i]
        leaves = filter_leaves(tree, x_nom, x_cont)

        n_leaves = np.zeros(len(y_uvs), dtype=np.int64)
        tot_leaf_weight = np.zeros(len(y_uvs), dtype=np.float32)
        max_leaf_weight = 0.0
        tot_w_ext_prob = np.zeros(len(y_uvs), dtype=np.float32)
        tot_exts = np.zeros(len(y_uvs), dtype=np.float32)
        tot_samples = np.zeros(len(y_uvs), dtype=np.float32)
        leaf_density = np.zeros(len(y_uvs), dtype=np.float32)

        zz_leaf_probs = np.zeros((len(leaves), len(y_uvs)), dtype=np.float32)
        zz_leaf_density = np.zeros((len(leaves), len(y_uvs)), dtype=np.float32)
        # tot_y = np.zeros(len(y_uvs), dtype=np.int32)
        for k, leaf in enumerate(leaves):
            spec_ext, ext_ws, L, ext_weight = stand.spec_exts[leaf.index]
            n_samples = len(leaf.sample_inds)
            leaf_weight = 1/(1.0+lam/n_samples)


            max_leaf_weight = max(leaf_weight, max_leaf_weight)


            
            # print("LEAF:", leaf.index, "L=", len(leaf.sample_inds), leaf_weight)
            # for enc_split, ext_w in zip(spec_ext, ext_ws):
            #     is_cont, negated, split, val = decode_split(enc_split)
            #     print(f"[{split}]=={val}", ext_w)

            y = np.argmax(leaf.counts)

            tot_samples[y] += n_samples
            ext_size, n_ext_matches, n_ext_fails, w_ext_matches, w_ext_fails = (
                eval_specific_extension(stand, leaf, x_nom, x_cont))
            # ext_prob = w_ext_matches / (w_ext_matches+w_ext_fails) if (w_ext_matches+w_ext_fails) > 0.0 else 1.0
            ext_prob = w_ext_matches / (n_ext_matches + n_ext_fails) if (n_ext_matches + n_ext_fails) > 0.0 else 1.0
            tot_w_ext_prob[y] += ext_prob
            tot_exts[y] += (n_ext_matches + n_ext_fails)
            # tot_exts[y] += n_ext_matches #/ (n_ext_matches + n_ext_fails)

            y_density[i][y] += leaf_weight * w_ext_matches
            probs[i][y]     += leaf_weight * ext_prob
            zz_leaf_probs[k][y] = ext_prob
            zz_leaf_density[k][y] = leaf_weight* ext_prob
            # probs[i][y] += n_ext_matches/(n_ext_matches+n_ext_fails) if ext_size > 0 else 1.0
            n_leaves[y] += 1
            tot_leaf_weight[y] += leaf_weight
            # print(f"LEAF: {y} {leaf.index} {n_samples}\t", ext_prob, w_ext_matches, w_ext_fails)
            # print(i, y, ":", w_ext_matches/(w_ext_matches+w_ext_fails), w_ext_matches, w_ext_fails)


        best_ind = np.argmax(probs[i])
        
        # print("probs", probs[i], best_ind, best_p)

        for j, y_class in enumerate(y_uvs):
            # if(j == best_ind):
            #     out_probs[i,best_ind] += probs[i,j]
            # else:
            #     out_probs[i,best_ind] += probs[i,j]

            if(n_leaves[j] > 0):
                # probs[i][j] /= tot_leaf_weight[j]
                # probs[i][j] /= tot_samples[j]
                y_density[i][j] / tot_samples[j]
                # probs[i][j] /= tot_leaf_weight[j]
                probs[i][j] /= np.sum(tot_leaf_weight)
                # probs[i][j] /= np.sum(tot_leaf_weight)

        # probs[i] /= np.sum(tot_leaf_weight)
        best_p = probs[i, best_ind]
        probs[i] = 1.0-best_p
        probs[i, best_ind] = best_p

        # print("out_probs", out_probs[i], best_p)

        # probs[i] = probs[i] / n_leaves

        # probs[i] = probs[i] / tot_leaf_weight # Normalize
        # probs[i] = probs[i] / np.sum(probs[i]) # Normalize
        # probs[i] = probs[i] / tot_exts # Normalize

        # probs[i] = (n_leaves / np.sum(n_leaves))
        # if(np.sum(n_leaves != 0) > 1):

        #################
        # print()
        # zz_max = np.empty((1,len(y_uvs)), dtype=np.float32)

        for j in range(len(y_uvs)):
            # print(j, "y[p]", -np.sort(-zz_leaf_probs[:,j]))
            # print(j, "y[d]", -np.sort(-zz_leaf_density[:,j]))
            max_k = np.argmax(zz_leaf_probs[:,j])
            zz_max[i,j] = zz_leaf_probs[max_k,j] #/ w_root #max_leaf_weight if max_leaf_weight != 0.0 else 1.0
            # print("BEST:", i, -np.sort(-zz_leaf_probs[:,i][zz_leaf_probs[:,i]>=zz_max[0,i]*.9]))

        # probs[i] = np.sum(zz_leaf_probs, axis=0)/tot_leaf_weight
        # probs[i] = np.sum(probs[i]*probs[i], axis=0)/np.sum(probs[i], axis=0)

        # best_probs = 
        # print(best_probs)



        #################


        
        # print("SQR_PROBS:", np.sum(zz_leaf_probs*zz_leaf_probs, axis=0)/np.sum(zz_leaf_probs, axis=0))
        # print("MAX_PROBS:", np.max(zz_leaf_probs[:,0]), np.max(zz_leaf_probs[:,1]))
        # print("PROBS    :", probs[i])
        # print("N_LEAVES :", n_leaves)

        
        # print("WEXTP:", tot_w_ext_prob / n_leaves)
        # print(" EXTP:", tot_exts / n_leaves)

    # b_ind = np.argmax(probs)

    # return y_density, probs
    return y_density, probs

@njit(cache=True)
def stand_predict_prob(stand, X_nom, X_cont):
    tree = stand.op_tree
    y_uvs = tree.data_stats.u_ys
    y_density, probs = stand_predict_y_density(stand, X_nom, X_cont)
    return probs, y_uvs

@njit(cache=True)
def instance_certainty(stand, X_nom, X_cont):
    tree = stand.op_tree
    L = max(len(X_nom),len(X_cont))
    if(len(X_nom) == 0): X_nom = np.empty((L,0), dtype=np.int32)
    if(len(X_cont) == 0): X_cont = np.empty((L,0), dtype=np.float32)
    
    y_uvs = tree.data_stats.u_ys

    # out = np.zeros((L,len(y_uvs)),dtype=prob_item_type)
    probs = np.zeros((L,len(y_uvs)),dtype=np.float64)
    # For each sample i, filter it into leaves and compute
    #  the probability of correctness on the basis of the specific extension 
    for i in range(L):
        x_nom, x_cont = X_nom[i], X_cont[i]
        leaves = filter_leaves(tree, x_nom, x_cont)

        # for j, y_class in enumerate(y_uvs):
        #     labels[i][j] = y_class
        n_leaves = np.zeros(len(y_uvs), dtype=np.int64)
        for leaf in leaves:

            n_branches = _count_covering_branches(tree, leaf, x_nom, x_cont)
            # The number of parent branches leading into that 
            #  don't (_nn) and do (_np) select (x_nom, x_cont)
            nn_np = n_branches[leaf.index]
            n_gen_fails, n_gen_matches = nn_np[0], nn_np[1]

            ext_size, n_ext_matches, n_ext_fails, w_ext_matches, w_ext_fails = (
                eval_specific_extension(stand, leaf, x_nom, x_cont))
            # print("G:", n_gen_matches, "/", n_gen_matches+n_gen_fails, "S:", n_ext_matches, "/", n_ext_matches+n_ext_fails)
            # print("log G:", np.log(1+n_gen_matches), "/", np.log(1+n_gen_matches+n_gen_fails))
            y = np.argmax(leaf.counts)

            den = (n_gen_matches+n_gen_fails)+(n_ext_matches+n_ext_fails)
            probs[i][y] += ((n_gen_matches)+(n_ext_matches))/den if den > 0 else 1.0
            n_leaves[y] += 1
            # den = np.log(1+n_gen_matches+n_gen_fails)+np.log(1+n_ext_matches+n_ext_fails)
            # probs[i][y] += (np.log(1+n_gen_matches)+np.log(1+n_ext_matches))/den if den > 0 else 1.0
            # n_log_n = lambda x: x * np.log(x)
            # den = np.log(1+n_gen_matches+n_gen_fails)+n_log_n(1+n_ext_matches+n_ext_fails)
            # probs[i][y] += (np.log(1+n_gen_matches)+n_log_n(1+n_ext_matches))/den if den > 0 else 1.0

            # gen = (1+n_gen_matches)/(1+n_gen_matches+n_gen_fails)
            # ext = (1+n_ext_matches)/(1+n_ext_matches+n_ext_fails)
            # probs[i][y] += (gen+ext)/2

        for j, y_class in enumerate(y_uvs):
            if(n_leaves[j] > 0):
                probs[i][j] /= n_leaves[j]
        # probs[i] /= n_leaves
    # print("PROBS:", probs)

    return probs, y_uvs


@njit(cache=True)
def instance_ambiguity(stand, x_nom, x_cont):

    tree = stand.op_tree
    if(stand.positive_class not in tree.data_stats.y_map):
        return 0.0
    pc = tree.data_stats.y_map[stand.positive_class]
    # nom_v_maps = tree.data_stats.nom_v_maps
    
    leaves = filter_leaves(tree, x_nom, x_cont)

    A_px = 0
    A_nx = 0
    Nn, Np = 0, 0
    for leaf in leaves:
        # print(leaf.counts)
        n_branches = _count_covering_branches(tree, leaf, x_nom, x_cont)
        # The number of parent branches leading into that 
        #  don't (_nn) and do (_np) select (x_nom, x_cont)
        nn_np = n_branches[leaf.index]
        _nn, _np = nn_np[0], nn_np[1]
        Nn += _nn
        Np += _np

        # Positive leaf case
        if(leaf.counts[pc] > 0):
            # Find the number of conditions failed in the specific extension 
            ext_size, n_ext_matches, n_ext_fails, w_ext_matches, w_ext_fails = (
                eval_specific_extension(stand, leaf, x_nom, x_cont))
            n_ext_failed = ext_size-n_ext_match

            A_px += _nn * (1 + ext_size)
            A_px += _np * (n_ext_fails)

            # print(_nn, ext_size, _np, n_ext_failed)
        

        # Negative Leaf case 
        else:
            A_nx += _np * (1+ext_size)

        # print("??", A_px, A_nx)
    
    # print(":", A_px, A_nx, Np, Nn)
    
    den = (Np + Nn)
    if(den == 0): return 0.0

    p = Np / den
    # print(p, A_px, A_nx)
    return p*A_px + (1-p)*A_nx


        
        









