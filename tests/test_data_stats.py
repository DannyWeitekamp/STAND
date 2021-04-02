import numpy as np
import pytest
from numbaILP.data_stats import *

X_nom_contig = np.array([
    # 3 & 4 are contiguous but not iterative contiguous
    [0,0,0,3,7,0],
    [1,1,1,0,6,0],
    [2,0,2,2,5,0],
    [3,1,0,0,4,0],
    [4,0,1,1,3,0],
    [5,1,2,0,2,0],
    [6,0,0,0,1,0],
    [7,1,1,0,0,0],
    ],dtype=np.int32)

X_cont = np.empty((len(X_nom_contig),0), dtype=np.float32)

Y_contig = np.array([0,1,2,0,1,2,0,3], dtype=np.int32)
Y_unorder_contig = np.array([3,3,0,1,2,0,1,2], dtype=np.int32)


@njit(cache=False)
def do_init(X_nom, X_cont, Y, v_contig, y_contig):
    ds = DataStats_ctor(nom_v_contiguous=v_contig, y_contiguous=y_contig)
    err = reinit_datastats(ds, X_nom, X_cont, Y)
    return err, ds

@njit(cache=False)
def do_update(X_nom, X_cont, Y, v_contig, y_contig):
    ds = DataStats_ctor(nom_v_contiguous=v_contig, y_contiguous=y_contig, ifit_enabled=True)
    for i in range(len(Y)): 
        # print(i,X_nom[i],X_cont[i],Y[i])
        update_data_stats(ds, X_nom[i], X_cont[i], Y[i])
    return 0, ds


def do_it(iterative, as_py, X_nom, X_cont, Y, v_contig, y_contig):
    f = do_update if(iterative) else do_init
    if(as_py): f = f.py_func
    return f(X_nom, X_cont, Y, v_contig, y_contig)


def _test_contiguous(iterative, as_py):
    #Test that everything works when we tell it that everything is contiguous
    err, ds = do_it(iterative, as_py, X_nom_contig, X_cont, Y_contig, True, True)

    assert list(ds.y_counts) == [3,2,2,1]
    # print(ds.n_vals)
    assert list(ds.n_vals) == [8,2,3,4,8,1]
    assert ds.n_classes == 4
    assert len(ds.u_ys) == 4

    #Test that it doesn't fail in a segfaulty way when Y isn't actually contiguous
    Y_not_contig = (Y_contig+2).astype(np.int32)
    err, ds = do_it(iterative, as_py, X_nom_contig, X_cont, Y_not_contig, True, True)
    assert len(ds.y_counts) == 6
    assert ds.n_classes == 6

    #TODO: what about negative classes? 

    #Test that it doesn't fail in a segfaulty way when X_nom isn't actually contiguous
    X_nom_not_contig = (X_nom_contig+2).astype(np.int32)
    err, ds = do_it(iterative, as_py, X_nom_not_contig, X_cont, Y_contig, True, True)

    assert list(ds.y_counts) == [3,2,2,1]
    # print(ds.n_vals)
    # assert list(ds.n_vals) == [8,2,3,4,8,1]
    # assert np.array_equal(ds.X_nom[:,0], X_nom_contig[:,0])
    assert np.array_equal(ds.Y , Y_contig)

    #TODO: what about negative nom vals? 


def _test_not_contiguous(iterative, as_py): 
    ''' Test cases where the incomming nominal values and class 
        labels are not expected to be contiguous.  ''' 
    X_nom_not_contig = X_nom_contig+2
    Y_not_contig = Y_contig+2

    err, ds = do_it(iterative, as_py, X_nom_not_contig, X_cont, Y_not_contig, False, False)

    assert list(ds.y_counts) == [3,2,2,1]
    assert dict(ds.y_map) == {2: 0, 3: 1, 4: 2, 5: 3}
    # print(ds.n_vals)
    assert list(ds.n_vals) == [8,2,3,4,8,1]

    assert dict(ds.nom_v_maps[0]) == {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
    assert dict(ds.nom_v_maps[-1]) == {2: 0}
    # print(ds.X_nom)
    assert np.array_equal(ds.X_nom[:,0], X_nom_contig[:,0])
    assert np.array_equal(ds.Y , Y_contig)
    assert ds.n_classes == 4
    assert len(ds.u_ys) == 4


 
def test_fit_contiguous():
    _test_contiguous(False, False)
    _test_contiguous(False, True)

def test_fit_not_contiguous():
    _test_not_contiguous(False, False)
    _test_not_contiguous(False, True)


def test_ifit_contiguous():
    _test_contiguous(True, False)
    _test_contiguous(True, True)

def test_ifit_not_contiguous():
    _test_not_contiguous(True, False)
    _test_not_contiguous(True, True)




# def test_update_contiguous(): 
#     err, ds = do_update(X_nom_contig, X_cont, Y_contig, True, True)
#     err1, ds1 = do_init.py_func(X_nom_contig, X_cont, Y_contig, True, True)

#     print(list(ds.y_counts), ds.n_vals)

#     assert err == err1
#     assert err == 0

#     assert list(ds.y_counts) == [3,2,2,1]
#     assert list(ds1.y_counts) == [3,2,2,1]

#     assert list(ds.n_vals) == [8,2,3,4,8,1]
#     assert list(ds1.n_vals) == [8,2,3,4,8,1]

 



if(__name__ == "__main__"):
    print("START")
    test_fit_contiguous()
    test_fit_not_contiguous()
    test_ifit_contiguous()
    test_ifit_not_contiguous()
    # test_update_contiguous()


