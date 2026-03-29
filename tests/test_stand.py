import numpy as np
from stand.stand import STANDClassifier

from test_tree_classifier import (
    make_data1, make_data2, make_data3,
    make_complex_continuous,
)


# ---------------------------------------------------------------------
# Helpers

def _fit(X_nom, X_cont, Y, **kwargs):
    stand = STANDClassifier(**kwargs)
    stand.fit(X_nom, X_cont, Y)
    return stand

def _fit_predict(X_nom, X_cont, Y, **kwargs):
    stand = _fit(X_nom, X_cont, Y, **kwargs)
    return stand, stand.predict(X_nom, X_cont)


# ---------------------------------------------------------------------
# Fit + Predict

def test_stand_fit_predict_nom():
    for data_gen in [make_data1, make_data2, make_data3]:
        X, Y = data_gen()
        X_nom = X.astype(np.int32)
        _, preds = _fit_predict(X_nom, None, Y)
        assert np.all(preds == Y), \
            f"{data_gen.__name__}: expected {Y}, got {preds}"


def test_stand_fit_predict_cont():
    X, Y = make_complex_continuous()
    _, preds = _fit_predict(None, X, Y)
    assert np.all(preds == Y), f"Expected {Y}, got {preds}"


# ---------------------------------------------------------------------
# predict_proba

def _check_predict_proba_shape(stand, X_nom, X_cont, N):
    """Verify probs has correct shape and non-negative values."""
    probs, y_uvs = stand.predict_proba(X_nom, X_cont)
    n_classes = len(y_uvs)
    assert probs.shape == (N, n_classes), \
        f"Expected probs shape ({N}, {n_classes}), got {probs.shape}"
    assert np.all(probs >= 0), \
        f"Probabilities must be non-negative:\n{probs}"
    return probs, y_uvs


def _proba_preds(probs, y_uvs):
    return np.array([y_uvs[i] for i in np.argmax(probs, axis=1)])


def test_predict_proba_nom_training():
    # predict_proba on training examples should assign highest prob to true label
    for data_gen in [make_data1, make_data2, make_data3]:
        X, Y = data_gen()
        X_nom = X.astype(np.int32)
        stand = _fit(X_nom, None, Y)
        probs, y_uvs = _check_predict_proba_shape(stand, X_nom, None, len(Y))
        preds = _proba_preds(probs, y_uvs)
        assert np.all(preds == Y), (
            f"{data_gen.__name__}: argmax(predict_proba) != Y\n"
            f"  y_uvs={y_uvs}  preds={preds}  Y={Y}"
        )


def test_predict_proba_cont_training():
    # predict_proba on the training set for the continuous dataset
    X, Y = make_complex_continuous()
    stand = _fit(None, X, Y)
    probs, y_uvs = _check_predict_proba_shape(stand, None, X, len(Y))
    print(probs)
    preds = _proba_preds(probs, y_uvs)
    assert np.all(preds == Y), (
        f"argmax(predict_proba) != Y\n"
        f"  y_uvs={y_uvs}  preds={preds}  Y={Y}"
    )


def test_predict_proba_cont_new_examples():
    # Train on make_complex_continuous, then evaluate predict_proba on
    # held-out examples that clearly resemble one class or the other.
    #
    # Training data recap:
    #   label 0: [7,1,2], [1,7,1], [6,7,7], [0,0,6]
    #   label 1: [5,8,1], [6,5,2], [7,7,1], [8,6,1]
    #
    # New label-1-like examples: high x0, high x1, low x2
    # New label-0-like examples: patterns that don't fit the above
    X_train, Y_train = make_complex_continuous()
    stand = _fit(None, X_train, Y_train)

    print(stand)

    X_test = np.array([
        # Expected label 1 (high x0+x1, low x2 — similar to training class 1)
        [9.0, 7.0, 1.0],
        [7.0, 5.0, 2.0],
        # Expected label 0 (low x1 or high x2 — similar to training class 0)
        [7.0, 1.0, 2.0],
        [0.0, 0.0, 7.0],
    ], dtype=np.float32)
    Y_expected = np.array([1, 1, 0, 0], dtype=np.int32)

    probs, y_uvs = _check_predict_proba_shape(stand, None, X_test, len(X_test))
    print(probs, y_uvs)
    preds = _proba_preds(probs, y_uvs)
    assert np.all(preds == Y_expected), (
        f"predict_proba on new examples gave wrong class\n"
        f"  y_uvs={y_uvs}  preds={preds}  expected={Y_expected}\n"
        f"  probs=\n{probs}"
    )

    assert np.max(probs[2] == 1.0)
    assert np.max(probs[0] < 1.0)
    assert np.max(probs[1] < 1.0)
    assert np.max(probs[3] < 1.0)



def test_predict_proba_matches_predict():
    # When pred_kind="prob", predict() should agree with argmax(predict_proba)
    X, Y = make_complex_continuous()
    stand = _fit(None, X, Y, pred_kind="prob")
    preds = stand.predict(None, X)
    probs, y_uvs = stand.predict_proba(None, X)
    proba_preds = _proba_preds(probs, y_uvs)
    assert np.all(preds == proba_preds), (
        f"predict() and argmax(predict_proba()) disagree\n"
        f"  predict()    = {preds}\n"
        f"  proba argmax = {proba_preds}"
    )


if __name__ == "__main__":
    test_predict_proba_cont_new_examples()
