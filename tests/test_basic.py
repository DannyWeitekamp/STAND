from stand.stand import STANDClassifier
import numpy as np

np.set_printoptions(precision=3)

stand = STANDClassifier(split_choice="all_near_max")


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
   # 1, 2, 3, 4, 5, 6, 7,    y
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

stand.fit(X,None,y)

print(stand.__str__(leaf_inds=True, node_inds=True))

# A case where it is distributed across two negative leaves
stand.predict_prob(np.array(
   [[1, 1, 1, 1, 0, 1, 1, 1]] # similar to 0 w/ [0] and [6] flipped
   ,dtype=np.int32
   ),None
)

print("--------------------------")

# A case where it is distributed across a positive and negative leaf
stand.predict_prob(np.array(
   [[1, 1, 1, 1, 1, 1, 0, 1]] # similar to 3 w/ [0] and [3] flipped
   ,dtype=np.int32
   ),None
)
