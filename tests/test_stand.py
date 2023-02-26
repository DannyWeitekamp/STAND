import numpy as np
from numba import njit, prange
from stand.stand import calc_invariant_nom_mask 
from stand.tree_classifier import TreeClassifier, encode_split, decode_split
from numba.core.runtime.nrt import rtsys





# def test_b_calc_invariant_nom_mask(benchmark):
#     benchmark.pedantic(run_tree_ifit,
#         setup=lambda : setup_tree_ifit('option_tree', random_XY, N=N),
#         warmup_rounds=1, rounds=10)
#     stats = benchmark.stats.stats
#     stats.data = [x/N for x in stats.data]

@njit(cache=True)
def calc_invariant_nom_mask(X_nom):
    nom_invariants = np.ones(X_nom.shape[1],dtype=np.uint8)
    x0 = X_nom[0]
    for i in range(1,len(X_nom)):
        nom_invariants &= (x0 == X_nom[i])
    return nom_invariants




if __name__ == "__main__":
    from cre.utils import PrintElapse

    x = np.ones((1000,300), dtype=np.int32)

    print(calc_invariant_nom_mask(x))

    with PrintElapse("calc_invariant_nom_mask"):
        calc_invariant_nom_mask(x)
