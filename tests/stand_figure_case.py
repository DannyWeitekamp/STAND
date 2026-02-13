from stand.stand import STANDClassifier
from stand.tree_classifier import SeqCovClassifier, TreeClassifier, opt_conjs_str

import numpy as np

def inv_mapper(inp_key, inp_val):
   return False, int(inp_key) + 1, inp_val


dt = TreeClassifier(
   inv_mapper=inv_mapper
)

stand = STANDClassifier(
   lam_p=25.0,
   # lam_e=25.0,
   slip=.5,
   impurity_func="gini",
   inv_mapper=inv_mapper,
   # fit_method="sequential_cover",
   split_choice="all_near_max",
   w_path_slip=True
   )

# # Original Figure
# Xy = np.array([
# #    # 1, 2, 3, 4, 5, 6, 7, y
#       [0, 0, 1, 1, 0, 1, 1, 0],#0
#       [1, 0, 0, 0, 1, 0, 1, 1],#1
#       [1, 1, 0, 1, 1, 1, 1, 1],#2
#       [1, 0, 1, 0, 0, 0, 0, 1],#3
#       [1, 1, 1, 1, 1, 1, 1, 1],#4
#       [1, 0, 0, 1, 0, 1, 1, 0],#5
#       [1, 1, 0, 1, 0, 1, 1, 0],#6
# ], dtype=np.int32)


# Pretty good, just too many early leaves
# Xy = np.array([
#    # 1, 2, 3, 4, 5, 6, 7, y
#     [0, 0, 1, 1, 0, 1, 1, 0], #0 *
#     [1, 1, 1, 0, 1, 0, 1, 1], #1 
#     [1, 1, 0, 1, 1, 1, 1, 1], #2
#     [1, 0, 0, 0, 0, 0, 0, 1], #3 *
#     [1, 1, 1, 1, 1, 1, 1, 1], #4
#     [1, 0, 1, 1, 0, 1, 1, 0], #5 *
#     [1, 1, 1, 0, 0, 1, 1, 0], #6 *
#     # [1, 1, 0, 1, 0, 1, 1, 0], #7
# ], dtype=np.int32)

Xy = np.array([
   # 1, 2, 3, 4, 5, 6, 7, y
    [0, 1, 1, 0, 1, 1, 0, 0], #0
    [1, 1, 0, 1, 0, 1, 1, 1], #1
    [1, 0, 1, 1, 1, 1, 1, 1], #2
    [1, 1, 0, 0, 0, 0, 0, 1], #3
    [1, 1, 1, 1, 1, 1, 1, 1], #4
    [1, 1, 1, 0, 1, 1, 0, 0], #5
    [1, 1, 1, 0, 1, 1, 1, 0], #6
    # [1, 1, 1, 1, 1, 0, 0, 0], #7
    # [1, 1, 1, 1, 0, 1, 1, 1], #8
], dtype=np.int32)




def print_specific_ex(Xy, inds):
   sub = Xy[inds][:,:-1]
   if(len(inds) == 1):
      all_same = np.ones(len(sub[0]))
   else:
      all_same = sub[0] == sub[1]
   for x in sub[2:]:
      all_same &= (x == sub[0])
   
   arr = []
   for i, x in enumerate(all_same):
      if(x):
         if(sub[0,i] == 1):
            arr.append(f"X{i+1}")
         else:
            arr.append(f"¬X{i+1}")
   return f"specific ext:{inds}: {', '.join(arr)}"

print(print_specific_ex(Xy, [3]))
print(print_specific_ex(Xy, [0,5,6]))
print(print_specific_ex(Xy, [1,3]))
print(print_specific_ex(Xy, [1,2,4]))
print(print_specific_ex(Xy, [0,5,6]))
print(print_specific_ex(Xy, [2,4]))
print(print_specific_ex(Xy, [6]))


   

X,y = Xy[:,:-1], Xy[:,-1]

for i,x0 in enumerate(X):
   for j,x1 in enumerate(X):
      if(i == j): 
         continue
      assert(np.array_equal(x0,x1), f"x0: {x0}, x1: {x1}")
# y = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)


# seq_cov = SeqCovClassifier()
# seq_cov.fit(X,None,y)
# print(seq_cov.__str__(leaf_inds=True, node_inds=True))

# raise ValueError()

import re
def easy_print(s):

   pf_map = {}
   for i in range(11):
      try:
         match = re.search(r'(NODE|ROOT|LEAF)\(' +str(i)+'\)', s)
         print(match, match.group(0), match.group(1), i)
         pf_map[str(i)] = match.group(1)[0]
      except:
         pass

   print(pf_map)

   matches = re.finditer(r'\(\[(\d+)\]==(\d+)\)\[F:(\d+) T:(\d+)\]', s)
   for match in matches:
      x = match.group(0)
      ft = match.group(1)
      val = match.group(2)
      false_dest = match.group(3)
      true_dest = match.group(4)
      l_dest, r_dest = (false_dest, true_dest) if val=="1" else (true_dest, false_dest)
      l_dest, r_dest = f"{pf_map[l_dest]}{l_dest}", f"{pf_map[r_dest]}{r_dest}"
      s = s.replace(x, f"¬X{ft}→ {l_dest}, X{ft}→ {r_dest}")
      # print(f"-X{ft} -> {l_dest}, -X{ft} -> {r_dest}")

   return s

def easy_print_opts(s):
   matches = re.finditer(r'(~*)\((\d+)==(\d+)\)', s)
   for match in matches:
      x = match.group(0)
      neg = match.group(1)
      ft = int(match.group(2)) +1
      val = match.group(3)
      invert = len(neg) > 0 ^ (val == "0")
      if(invert):
         s = s.replace(x, f"¬X{ft}")
      else:
         s = s.replace(x, f"X{ft}")
   return s

dt.fit(X,None,y)
stand.fit(X,None,y)

print("------  STAND  -------")
print(stand.__str__(leaf_inds=True, node_inds=True))
print(easy_print(stand.__str__(leaf_inds=True, node_inds=True)))
opt_conjs = stand.get_opt_conjs_for_label(1)
print(easy_print_opts(opt_conjs_str(stand.op_tree, opt_conjs)))

print("-------  DT  --------")
print(dt.__str__(leaf_inds=True, node_inds=True))
print(easy_print(dt.__str__(leaf_inds=True, node_inds=True)))
opt_conjs = dt.get_opt_conjs_for_label(1)
print(opt_conjs_str(dt.tree, opt_conjs))

#              1,  2,  3,  4,  5,  6,  7
nom_weights = [1., 1., 5., 2., 1., 1., 1.]
stand.fit(X,None,y,nom_ft_weights=np.array(nom_weights))

print("------  STAND WEIGHTED  -------")
print(stand.__str__(leaf_inds=True, node_inds=True))
print(easy_print(stand.__str__(leaf_inds=True, node_inds=True)))
opt_conjs = stand.get_opt_conjs_for_label(1)
print(easy_print_opts(opt_conjs_str(stand.op_tree, opt_conjs)))
