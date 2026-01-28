from stand.stand import STANDClassifier
from stand.tree_classifier import SeqCovClassifier, TreeClassifier
import numpy as np

def inv_mapper(inp_key, inp_val):
   return False, int(inp_key) + 1, inp_val


dt = TreeClassifier(
   inv_mapper=inv_mapper
)

stand = STANDClassifier(
   # lam_p=25.0,
   # lam_e=25.0,
   slip=.5,
   impurity_func="gini",
   inv_mapper=inv_mapper,
   # fit_method="sequential_cover",
   split_choice="all_near_max",
   w_path_slip=True
   )


Xy = np.array([
   # 1, 2, 3, 4, 5, 6, 7, y
    [0, 0, 1, 1, 0, 1, 1, 0], #0
    [1, 1, 1, 0, 1, 0, 1, 1], #1
    [1, 1, 0, 1, 1, 1, 1, 1], #2
    [1, 0, 0, 0, 0, 0, 0, 1], #3
    [1, 1, 1, 1, 1, 1, 1, 1], #4
    [1, 0, 1, 1, 0, 1, 1, 0], #5
    [1, 1, 1, 1, 0, 0, 1, 0], #6
    # [1, 1, 0, 1, 0, 1, 1, 0], #7
], dtype=np.int32)


for i,x0 in enumerate(Xy[:1]):
   for j,x1 in enumerate(Xy[:1]):
      if(i == j): 
         continue
      assert(np.array_equal(x1,x2))

   

X,y = Xy[:,:-1], Xy[:,-1]
# y = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)


# seq_cov = SeqCovClassifier()
# seq_cov.fit(X,None,y)
# print(seq_cov.__str__(leaf_inds=True, node_inds=True))

# raise ValueError()





dt.fit(X,None,y)
stand.fit(X,None,y)

print("------  STAND  -------")
print(stand.__str__(leaf_inds=True, node_inds=True))

print("-------  DT  --------")
print(dt.__str__(leaf_inds=True, node_inds=True))

