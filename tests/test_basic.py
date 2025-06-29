from stand.stand import STANDClassifier
from stand.tree_classifier import SeqCovClassifier
import numpy as np

np.set_printoptions(precision=3)

stand = STANDClassifier(
   lam_p=25.0,
   lam_e=1.0,
   slip=.3,
   impurity_func="gini",
   # fit_method="sequential_cover",
   split_choice="all_near_max",
   w_path_slip=True
   )


# Xy = np.array([
#    # 1, 2, 3, 4, 5, 6, 7, y
#     [0, 0, 1, 1, 0, 1, 1, 0], #0
#     [1, 0, 0, 0, 1, 0, 1, 1], #1
#     [1, 1, 0, 1, 1, 1, 1, 1], #2
#     [1, 0, 1, 0, 0, 0, 0, 1], #3
#     [1, 1, 1, 1, 1, 1, 1, 1], #4
#     [1, 0, 0, 1, 0, 1, 1, 0], #5
#     [1, 1, 0, 1, 0, 1, 1, 0], #6
#     # [1, 1, 0, 1, 0, 1, 1, 0], #7
# ], dtype=np.int32)

Xy = np.array([
   # 0, 1, 2, 3, 4, 5, 6, 7  y
    [0, 1, 1, 1, 0, 1, 1, 1, 0], #0 *
    [0, 0, 0, 1, 0, 1, 0, 1, 0], #1 *
    [1, 0, 1, 1, 1, 1, 1, 1, 1], #2 *
    [0, 1, 0, 1, 1, 1, 0, 1, 1], #3 *
    [1, 0, 1, 0, 0, 0, 1, 1, 1], #4 
    [0, 0, 0, 1, 0, 1, 1, 1, 0], #5 *
    [0, 0, 1, 0, 0, 0, 0, 1, 1], #6 
    [0, 0, 0, 1, 0, 1, 1, 1, 0], #7 *
    [0, 0, 1, 0, 0, 0, 1, 1, 1], #8 
    # [0, 1, 1, 1, 0, 1, 1, 1, 0], #0 *
    # [0, 0, 0, 1, 0, 1, 0, 1, 0], #1 *
    # [1, 0, 1, 1, 1, 1, 1, 1, 1], #2 *
    # [0, 1, 0, 1, 1, 1, 0, 1, 1], #3 *
    # [1, 0, 1, 0, 0, 0, 1, 1, 1], #4 
    # [0, 0, 0, 1, 0, 1, 1, 1, 0], #5 *
    # [0, 0, 1, 0, 0, 0, 0, 1, 1], #6 
    # [0, 0, 0, 1, 0, 1, 1, 1, 0], #7 *
    # [0, 0, 1, 0, 0, 0, 1, 1, 1], #8 
], dtype=np.int32)

X,y = Xy[:,:-1], Xy[:,-1]
# y = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)


# seq_cov = SeqCovClassifier()
# seq_cov.fit(X,None,y)
# print(seq_cov.__str__(leaf_inds=True, node_inds=True))

# raise ValueError()

stand.fit(X,None,y)
stand.fit(X,None,y)

print(stand.__str__(leaf_inds=True, node_inds=True))
print(stand.get_conds(1))

# A case where it is distributed across two negative leaves
stand.predict_proba(np.array(
   [[1, 1, 1, 1, 0, 1, 1, 1]] # similar to 0 w/ [0] and [6] flipped
   ,dtype=np.int32
   ),None
)

print("--------------------------")

# A case where it is distributed across a positive and negative leaf
probs, labels = stand.predict_proba(np.array(
    # [0, 1, 0, 1, 1, 1, 0, 1, 1]
   [[1, 1, 1, 1, 1, 1, 0, 1]] # similar to 3 w/ [0] and [3] flipped
   ,dtype=np.int32
   ),None
)
print(probs)

print("--------------------------")

# A case where only goes to positive leaf
probs, labels = stand.predict_proba(np.array(
    # [0, 1, 0, 1, 1, 1, 0, 1, 1]
   [[0, 1, 0, 1, 1, 1, 0, 1]] # Exactly same as 3
   ,dtype=np.int32
   ),None
)
print(probs)

print("--------------------------")

# A case where only goes to negative leaf
probs, labels = stand.predict_proba(np.array(
    # [0, 1, 0, 1, 1, 1, 0, 1, 1]
   [[0, 1, 1, 1, 0, 1, 1, 1]] # similar to 3 w/ [3] and [5] flipped
   ,dtype=np.int32
   ),None
)

print(probs)



