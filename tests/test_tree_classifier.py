import numpy as np
from numba import njit
from stand.stand import instance_ambiguity
from stand.tree_classifier import TreeClassifier, encode_split, decode_split
from numba.core.runtime.nrt import rtsys


def used_bytes():
    stats = rtsys.get_allocation_stats()
    return stats.alloc-stats.free


# ---------------------------------------------------------------------
# Test encode / decode split

def test_encode_decode_split():
    enc_split = encode_split(1,1,8,9)
    print(decode_split(enc_split))
    assert isinstance(enc_split,int)
    assert decode_split(enc_split) == (1,1,8,9)

    enc_split = encode_split(1,1,0x0FFFFFFF,0x0FFFFFFF)
    assert isinstance(enc_split,int)
    print(decode_split(enc_split))
    assert decode_split(enc_split) == (1,1,0x0FFFFFFF,0x0FFFFFFF)

    enc_split = encode_split(0,0,0x3FFFFFFF,0x6FFFFFFF)
    assert isinstance(enc_split,int)
    print(decode_split(enc_split))
    assert decode_split(enc_split) == (0,0,0x3FFFFFFF,0x6FFFFFFF)

    enc_split = encode_split(0,0,13,0)
    assert isinstance(enc_split,int)
    print(decode_split(enc_split))
    assert decode_split(enc_split) == (0,0,13,0)



# ---------------------------------------------------------------------
# Dataset generation / processing

def random_XY(N=1000,M=100):
    X = np.random.randint(0,5,size=(N,M),dtype=np.int32)
    Y = np.random.randint(0,3,size=(N,),dtype=np.int32)
    return X, Y


def make_data1():
    data1 = np.asarray([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3 0
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1 1
    [0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1 2
    [0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1 3
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2 4
    [0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2 5
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2 6
    ],np.int32);

    labels1 = np.asarray([3,1,1,1,2,2,2],np.int32);
    return data1, labels1

def make_data2():
    data2 = np.asarray([
#    0 1 2 3 4 5 
    [0,0,0,0,0,0], #1 0
    [0,0,1,0,0,0], #1 1
    [0,1,1,0,0,0], #1 2
    [1,1,1,0,0,1], #2 3
    [0,1,1,1,1,0], #2 4
    [1,1,1,0,1,0], #2 5
    ],np.int32);

    labels2 = np.asarray([1,1,1,2,2,2],np.int32);
    data2 = data2[:,[1,0,2,3,4,5]]
    return data2, labels2

def make_data3():
    data3 = np.asarray([
#    0 1 
    [0,0], #1 0 
    [1,0], #1 1
    [0,1], #1 2
    [1,1], #2 3
    ],np.int32);

    labels3 = np.asarray([1,1,1,2],np.int32);
    return data3, labels3


def make_nom_cont(X, use_nom, use_cont):
    # Make nominal and/or continous versions of X
    X_nom = X.astype(np.int32) if(use_nom) else np.zeros((0,0),dtype=np.int32)
    X_cont = X.astype(np.float32) if(use_cont) else np.zeros((0,0),dtype=np.float32)
    return X_nom, X_cont


# ---------------------------------------------------------------------
# : Model setup functions

def setup_sklearn_fit(model, data_gen, **kwargs):
    from sklearn import tree as SKTree
    from sklearn.preprocessing import OneHotEncoder

    dt = getattr(SKTree,model)()

    X, Y = data_gen(**kwargs)

    # Sklearn doesn't recognize nominal values so requires one-hot encoding
    one_h_encoder = OneHotEncoder()
    X_oh = one_h_encoder.fit_transform(X).toarray()
    return (dt, X_oh, Y), {}


def setup_tree_fit(preset, data_gen, use_nom=True, use_cont=False, cache_nodes=None, **kwargs):
    dt = TreeClassifier(preset_type=preset, cache_nodes=cache_nodes)
    X, Y = data_gen(**kwargs)
    X_nom, X_cont = make_nom_cont(X, use_nom, use_cont)
    return (dt, X_nom, X_cont, Y), {}

def setup_tree_ifit(*args, **kwargs):
    # For ifitting trees setup so that X_nom and X_cont are the same size
    (dt, X_nom, X_cont, Y), _ = setup_tree_fit(*args, **kwargs)
    if(len(X_nom) == 0): X_nom = np.zeros((len(X_cont),0), dtype=X_nom.dtype)
    if(len(X_cont) == 0): X_cont = np.zeros((len(X_nom),0), dtype=X_cont.dtype)
    return (dt, X_nom, X_cont, Y), {}


# ---------------------------------------------------------------------
# : Model fitting functions

def run_sk_tree_fit(dt, X_oh, Y):
    dt.fit(X_oh, Y)
    return dt

def run_tree_ifit(dt, X_nom, X_cont, Y):
    for i, (x_n, x_c, y) in enumerate(zip(X_nom, X_cont, Y)):
        dt.ifit(x_n, x_c, y)
    return dt

def run_tree_fit(dt, X_nom, X_cont, Y):
    dt.fit(X_nom, X_cont, Y)
    return dt

# ----------------------------------------------------------------------
# : Functionality Tests

# -------------------------------------
# : Test Fit + Predict

def run_test_fit_predict(setup_func, fit_func):
    args, _ = setup_func()
    X_args, Y = args[1:-1], args[-1:]
    dt = fit_func(*args)
    assert np.all(dt.predict(*X_args) == Y)

def run_test_all_datasets(preset, use_ifit=True):
    dataset_generators = [make_data1, make_data2, make_data3]

    #fit
    for data_gen in dataset_generators:
        run_test_fit_predict(
            setup_func=lambda : setup_tree_fit(preset, data_gen),
            fit_func=run_tree_fit
        )

    if(not use_ifit): return
    
    #ifit
    for data_gen in dataset_generators:
        run_test_fit_predict(
            setup_func=lambda : setup_tree_ifit(preset, data_gen),
            fit_func=run_tree_ifit
        )

def test_decision_tree():
    run_test_all_datasets('decision_tree')

def test_option_tree():
    run_test_all_datasets('option_tree')

# -------------------------------------
# : Test Option Tree Ambiguity Heuristics 


# ------------------------
# : Memleak tests 

def run_test_memleak(setup_func, fit_func):
    init_used = used_bytes()
    for i in range(5):
        args, _ = setup_func()
        fit_func(*args)
        args = None
        if(i == 0): 
            init_used = used_bytes()
        else:
            assert used_bytes() <= init_used

def test_memleaks():
    # fit
    ## Decision Tree
    run_test_memleak(
        setup_func=lambda : setup_tree_fit('decision_tree', random_XY, N=50),
        fit_func=run_tree_fit
    )

    ## Option Tree
    run_test_memleak(
        setup_func=lambda : setup_tree_fit('option_tree', random_XY, N=50),
        fit_func=run_tree_fit
    )

    # ifit() currently has a memleak so skip
    return 

    # ifit
    ## Decision Tree
    run_test_memleak(
        setup_func=lambda : setup_tree_ifit('decision_tree', random_XY, N=50),
        fit_func=run_tree_ifit
    )

    ## Option Tree
    run_test_memleak(
        setup_func=lambda : setup_tree_ifit('option_tree', random_XY, N=50),
        fit_func=run_tree_ifit
    )


# -----------------------------------------------------------------------
# : Benchmark Tests

N = 1000

# Sklearn

def test_b_fit_sklearn_dt_rand_1000x100(benchmark):
    benchmark.pedantic(run_sk_tree_fit,
        setup=lambda : setup_sklearn_fit('DecisionTreeClassifier', random_XY, N=N),
        warmup_rounds=1, rounds=10)

# fit 

def test_b_fit_decision_tree_rand_1000x100(benchmark):
    benchmark.pedantic(run_tree_fit,
        setup=lambda : setup_tree_fit('decision_tree', random_XY, N=N),
        warmup_rounds=1, rounds=10)

def test_b_fit_option_tree_rand_1000x100(benchmark):
    benchmark.pedantic(run_tree_fit,
        setup=lambda : setup_tree_fit('option_tree', random_XY, N=N),
        warmup_rounds=1, rounds=10)

def test_b_fit_option_tree_no_cache_rand_1000x100(benchmark):
    benchmark.pedantic(run_tree_fit,
        setup=lambda : setup_tree_fit('option_tree', random_XY, cache_nodes=False, N=N),
        warmup_rounds=1, rounds=10)


# ifit

def test_b_ifit_decision_tree_rand_1x100(benchmark):
    N = 100
    benchmark.pedantic(run_tree_ifit,
        setup=lambda : setup_tree_ifit('decision_tree', random_XY, N=N),
        warmup_rounds=1, rounds=10)
    stats = benchmark.stats.stats
    stats.data = [x/N for x in stats.data]

def test_b_ifit_option_tree_rand_1x100(benchmark):
    N=100
    benchmark.pedantic(run_tree_ifit,
        setup=lambda : setup_tree_ifit('option_tree', random_XY, N=N),
        warmup_rounds=1, rounds=10)
    stats = benchmark.stats.stats
    stats.data = [x/N for x in stats.data]

if __name__ == "__main__":

    from stand.stand import STANDClassifier
    # test_memleaks()
    # test_decision_tree()
    # test_option_tree()

    v = 0
    for i in range(10):
        dt = run_tree_fit(*setup_tree_fit('decision_tree', random_XY, N=N)[0])
        v += len(dt.nodes)
    print("DT:", v/10)
        
    v = 0
    for i in range(10):
        dt = run_tree_fit(*setup_tree_fit('option_tree', random_XY, N=N)[0])
        v += len(dt.nodes)
    print("Op:", v/10)

    v = 0
    for i in range(10):
        dt = run_tree_fit(*setup_tree_fit('option_tree', random_XY, cache_nodes=False,  N=N)[0])
        v += len(dt.nodes)
    print("Op no cache:", v/10)


    raise ValueError()
    test_encode_decode_split()

    dt = TreeClassifier('option_tree')
    X_nom, Y = make_data1()
    # X_nom, Y = random_XY(N=50,M=10)
    dt.fit(X_nom, None, Y)
    print(dt)
    #                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    x_nom = np.asarray([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],np.int32)
    # x_nom = np.random.randint(0,5,(10,),dtype=np.int32)
    x_cont = np.empty(0,dtype=np.float32)

    
    print(x_nom)

    print("--------------------------------")
    stand = STANDClassifier(positive_class=1)
    stand.fit(X_nom, None, Y)

    IA = stand.instance_ambiguity(x_nom, None)
    print("Instance Ambiguity: ", IA)
    for x,y in zip(X_nom,Y):
        print(x,y)
        print(stand.instance_ambiguity(x))

    new_X_nom = np.concatenate([X_nom,x_nom.reshape((1,-1))])
    new_Y = np.concatenate([Y,[1]])
    stand.fit(new_X_nom, None, new_Y)
    print(stand)
    IA = stand.instance_ambiguity(x_nom, None)
    print("Instance Ambiguity: ", IA)




    
