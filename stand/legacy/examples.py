import numpy as np
from concept_formation.preprocessor import Tuplizer,Flattener
# from d_tree import 
from pprint import pprint
from d_tree import test_Afit, print_tree


def flatten_state(state):
    tup = Tuplizer()
    flt = Flattener()
    state = flt.transform(tup.transform(state))
    return state

def clean_and_flatten(X,ignore=[]):
	new_X = []
	for i,state in enumerate(X):
		new_state = {}
		for name,elem in state.items():
			new_elem = {}
			for attr, value in elem.items():
				if(attr not in ignore):
					new_elem[attr] = value
			new_state[name] = new_elem
		X[i] = new_state


	X = [flatten_state(x) for x in X]
	return X

def vectorize(X,dtype=None):
	d = {}
	n = 0
	for state in X:
		for k,v in state.items(): 
			if(k not in d):
				d[k] = n
				n += 1

	L = len(d.keys())
	out = np.zeros((len(X),2*L),dtype=dtype)
	for i,state in enumerate(X):
		for k,v in state.items():
			out[i,d[k]] = v
			out[i,d[k]+L] = 1

	return out

dir_map = {"to_left": "l", "to_right": "r", "above": "a", "below":"b", "offsetParent":"p"}
dirs = list(dir_map.keys())
def _relative_rename_recursive(state,center,center_name="sel",mapping=None,dist_map=None):
    if(mapping is None):
        mapping = {center:center_name}
        dist_map = {center:0}
    # print(state)
    center_obj = state[center]

    stack = []
    for d in dirs:
        ele = center_obj.get(d,None)
        # print("ele")
        # print(ele)
        if(ele is None or ele == "" or
          (ele in dist_map and dist_map[ele] <= dist_map[center] + 1) or
           ele not in state):
            continue
        mapping[ele] = center_name + "." + dir_map[d]
        dist_map[ele] = dist_map[center] + 1
        stack.append(ele)
    # pprint(mapping)
    for ele in stack:
        _relative_rename_recursive(state,ele,mapping[ele],mapping,dist_map)

    return mapping



# data = [{"A": 1, "B" : 2},{"C": 1, "B" : 2},{"C": 1, "D" : 2}]

# print(vectorize(data,ignore=['above','below', 'left', 'right']))

#A1, B1
#A2, B2

states = [
	#Start State
	{
	"A1" : {"l":0, "v":0},"B1" : {"l":1, "v":2},
	"A2" : {"l":0, "v":0},"B2" : {"l":0, "v":0},
	},
	#Unordered
	{
	"A1" : {"l":0, "v":0},"B1" : {"l":1, "v":2},
	"A2" : {"l":0, "v":0},"B2" : {"l":1, "v":3},
	},
	{
	"A1" : {"l":1, "v":4},"B1" : {"l":1, "v":2},
	"A2" : {"l":0, "v":0},"B2" : {"l":0, "v":0},
	},
	{
	"A1" : {"l":1, "v":4},"B1" : {"l":1, "v":2},
	"A2" : {"l":0, "v":0},"B2" : {"l":1, "v":3},
	},
	#Last Step
	{
	"A1" : {"l":1, "v":4},"B1" : {"l":1, "v":2},
	"A2" : {"l":1, "v":6},"B2" : {"l":1, "v":3},
	}

]

relations = {"A1" :{"above": None, "below": "A2",'to_left': None, 'to_right': "B1"},
			 "A2" :{"above": "A1", "below": None,'to_left': None, 'to_right': "B2"},
			 "B1" :{"above": None, "below": "B2",'to_left': "A1", 'to_right': None},
			 "B2" :{"above": "B1", "below": None,'to_left': "A2", 'to_right': None},
			}

for state in states:
	for e in state:
		state[e].update(relations[e])



all_bindings = [
	[('double',"A1","B1"),('add',"B2","B1")],
	[('double',"A1","B1"),('double',"A2","B2")],
	[('add',"B2","B1"),('double',"A2","B2")],
	[('double',"A2","B2")],
	[]
]
ground_truth = [
	[('double',"A1","B1"),('add',"B2","B1")],
	[('double',"A1","B1")],
	[('add',"B2","B1")],
	[('double',"A2","B2")],
	]


states_by_op = {}
lables_by_op = {}
for i,state in enumerate(states):
	all_states = []
	for binding in all_bindings[i]:
		print(binding)
		remap = _relative_rename_recursive(state,binding[1])
		r_state = {remap[k] : v for k,v in state.items()}
		# all_states.append(r_state)
		sl = states_by_op.get(binding[0],[])
		sl.append(r_state)
		states_by_op[binding[0]] = sl

		ll = lables_by_op.get(binding[0],[])
		ll.append(binding in ground_truth[i])
		lables_by_op[binding[0]] = ll

		pprint(r_state)

for op in states_by_op:
	states = states_by_op[op]
	labels = lables_by_op[op]
	flat_states = clean_and_flatten(states,ignore=['above','below', 'to_left', 'to_right'])
	v_state = vectorize(flat_states,np.uint8)
	print(v_state)
	print(labels)
	print(type(v_state))
	print_tree(test_Afit(v_state,np.array(labels,dtype=np.int32)))



# print(_relative_rename_recursive(states[0],'A1'))
	

# print(vectorize(states[0],ignore=['above','below', 'left', 'right']))

