#include <iostream>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xmasked_view.hpp"
// #include "xtensor/xmasked_value.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using xt::xarray;
using xt::keep;
using xt::all;
using xt::view;


#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
void die_handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

struct TreeType{
	bool isLeaf = false;
};


// typedef class Tree Tree;
struct Tree : TreeType{
	int split_on = -1;
	TreeType *left = (Tree *) NULL;
	TreeType *right = (Tree *) NULL;
	Tree(){
		isLeaf = false;
	}
};

struct Leaf : TreeType{
	vector<uint>* data;
	Leaf(){
		isLeaf = true;
	}
};

// #define COUNT 5

void print2DUtil(TreeType *in, int space)  
{  
	if(!in->isLeaf){
		Tree* root = (Tree *) in;
	    space += 5;  
	    if(root->right != NULL || root->left != NULL){
	    	cout << std::string(space, ' ')  << root->split_on << ":" << "\n";
	    	if(root->left != NULL){
	    		print2DUtil(root->left, space);  
	    	}
	    	if(root->right != NULL){
	    		print2DUtil(root->right, space);  	
	    	}
	    }else{
			cout << std::string(space, ' ')  << root->split_on << "\n";
	    }
	}else{
		cout << "[";
		vector<uint> *leaf_data = ((Leaf*)in)->data;
		for(uint i=0; i < (*leaf_data).size();i++){
			cout << (*leaf_data)[i];
			if(i != (*leaf_data).size()-1){
				cout << ", ";
			}
		}
		cout << "]";
		// cout << *((Leaf *) in)->data;
	}
  
    // Process right child first  
    
  
    // Print current Tree after space  
    // // count  
    // cout<<endl;  
    // for (int i = COUNT; i < space; i++)  
    //     cout<<" ";  
    // cout<<root->split_on<<"\n";  
  
    // Process left child  
    
}  
  
// Wrapper over print2DUtil()  
void print2D(Tree *root)  
{  
    // Pass initial space count as 0  
    print2DUtil(root, 0);  
}  

// void printTree(Tree *in, int level=0, string delim="    "){
// 	string num = in->split_on == -1 ? "*" else std::to_string(in.split_on);
// 	std::stringstream ss;
// 	for(size_t i = 0; i < v.size(); ++i){
// 	  if(i != 0)
// 	    ss << ",";
// 	  ss << v[i];
// 	}
// 	d = " ".join([chr(a+64)+":"+str(b) for a,b in sorted(self.counts.items(),key=lambda x: x[0])])
// }

// Tree *newTree


// template <class L>
xarray<double> gini(xarray<int> counts){
	xarray<double> totals = xt::expand_dims(xt::cast<double>(xt::sum(counts,1)) + 1e-10,1);
	// cout << totals << std::endl;
	xarray<double> prob = xt::cast<double>(counts) / totals;
	// cout << prob << std::endl;
	xarray<double> impurity = (1.0-xt::sum(xt::square(prob), 1));
	// cout << impurity << std::endl;
	return impurity;
}


std::tuple<xarray<uint>,xarray<int>,xarray<uint>> unique_counts(xarray<int> inp){
	vector<uint> counts;
	vector<int> uniques;
	xarray<uint> inds = xt::zeros<uint>(inp.shape());
	uint ind=0;
	uint last = 0;
	for(uint i=1; i < inp.size()+1;i++){
		if(i == inp.size() || inp[i-1] != inp[i]){
			counts.push_back(i-last);
			uniques.push_back(inp[i-1]);
			last = i;
			ind++;
		}
		inds[i] = ind;
	}
	return std::make_tuple(xt::adapt(counts),xt::adapt(uniques),inds);
}

xarray<uint> counts_per_split(xarray<uint> start_counts, xarray<bool> x, xarray<uint> y_inds){
	//(n, s)
	// xarray<size_t> size = 
	// cout << start_counts << std::flush;
	// cout << start_counts.size() << std::flush;
	xarray<uint> counts = xt::zeros<uint>({(size_t)x.shape()[1],(size_t)2,start_counts.size()});
	// auto zeros = (x == 0)
	
	for(uint i=0; i<x.shape()[0]; i++){
		for(uint j=0; j<x.shape()[1]; j++){
			// cout << i << ", " << j << ", " << y_inds[i] << std::endl <<std::flush;	
			if(x(i,j)){
				counts(j,1,y_inds[i]) += 1;	
			}else{
				counts(j,0,y_inds[i]) += 1;	
			}
		}
	}
	return counts;

}

// template <class Lambda>
TreeType *split_tree(xarray<bool> x, xarray<uint> y_inds,
			    double entropy,
			    xarray<double> (*entropy_func)(xarray<int>),
			    xarray<uint> start_counts){

	// cout << "C" << std::endl << std::flush;
	
	xarray<uint> counts = counts_per_split(start_counts,x,y_inds);
	xarray<uint> counts_l = view(counts,all(),0,all());
	xarray<uint> counts_r = view(counts,all(),1,all());
	xarray<double> entropy_l = entropy_func(counts_l);
	xarray<double> entropy_r = entropy_func(counts_r);
	xarray<double> entropy_lr = entropy_l + entropy_r;
	xarray<double> utility = entropy - (entropy_lr);
	uint max_split = xt::argmax(utility)[0];

	
	// cout << utility << std::endl;
	// cout << counts << std::endl;
	
	// void* out;
	if(utility[max_split] <= 0){
		xarray<uint> max_count = view(counts,max_split,0,all());
		vector<uint>* leaf_data = new vector<uint>(max_count.size());
		cout << "MEEP: " << max_count;
		for(uint i=0;i < max_count.size();i++){
			(*leaf_data)[i] = max_count[i];
		}
		Leaf *leaf = new Leaf();
		leaf->data = leaf_data; 
		return (TreeType *) leaf;
	}
	cout << max_split;

	Tree *left_Tree;	
	Tree *right_Tree;	
	if(entropy_lr[max_split] > 0){
		cout << ":";

		xarray<bool> mask = view(x,all(),max_split);
		// cout << "MASK" << mask << std::endl;
		if(entropy_l[max_split] > 0){
			// auto r = xt::arange(x.shape()[0]);
			auto sel_l = view(xt::from_indices(xt::argwhere(!mask)),all(),0);
			// cout << "SEL_L" << sel_l <<std::endl;
			// cout << "X" << view(x,keep(sel_l),all()) << std::endl;
			cout << " L(";
			left_Tree = (Tree *)split_tree(view(x,keep(sel_l),all()),
						 		   view(y_inds,keep(sel_l),all()),
						 entropy_l[max_split], gini,counts_l[max_split]);
			cout << ")";
		}else{
			left_Tree = new Tree();
		}

		if(entropy_r[max_split] > 0){
			auto sel_r = view(xt::from_indices(xt::argwhere(mask)),all(),0);
			// cout << "SEL_R" << sel_r << std::endl;

			cout << " R(";
			right_Tree = (Tree *)split_tree(view(x,keep(sel_r),all()),
						 		    view(y_inds,keep(sel_r),all()),
						 entropy_r[max_split], gini,counts_r[max_split]);
			cout << ")";
		}else{
			right_Tree = new Tree();
		}
		// if()
	}else{
		left_Tree = new Tree();
		right_Tree = new Tree();
	}
	Tree *out = new Tree();
	out->split_on = max_split;
	out->left = left_Tree;
	out->right = right_Tree;
	return (TreeType *)out;
}

// template <typename F>
void* binary_decision_tree(xarray<bool> x, xarray<int> y, xarray<double> (*entropy_func)(xarray<int>)){
	xarray<long unsigned int> sorted_inds = xt::argsort(y);
	auto x_sorted = view(x,keep(sorted_inds),all());
	auto y_sorted = view(y,keep(sorted_inds));
	cout << sorted_inds << std::endl;
	cout << x_sorted << std::endl;
	cout << y_sorted << std::endl;

	auto [counts, u_ys,y_inds] = unique_counts(y_sorted);
	cout << counts << ", " << u_ys << std::endl;
	cout << counts << ", " << y_inds << std::endl;
	auto entropy = entropy_func(counts.reshape({1,counts.size()}));
	cout << entropy << std::endl;
	return split_tree(x_sorted, y_inds, entropy[0], gini,counts);
}

void prediction_tree(xarray<bool> x, Tree *tree){



}

int main() 
{
	
	signal(SIGSEGV, die_handler); 
	xarray<bool> data = {
	//	 0 1 2 3 4 5 6 7 8 9 10111213141516
		{0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1}, //3
		{0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0}, //1
		{0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0}, //1
		{0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0}, //1
		{1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1}, //2
		{0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0}, //2
		{1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0}, //2
	};

	xarray<double> labels = {3,1,1,1,2,2,2};
	
	Tree* tree = (Tree*) binary_decision_tree(data,labels,gini);
	cout << endl;
	print2D(tree);

	// xarray<int> indicies = {0,2,4,5};
	// cout << view(data,keep(indicies),all()) << std::endl;
	// // cout << xt::index_view(data,{0,2,4,5}) << std::endl;

 //    xarray<double> arr1
 //      {1.0, 2.0, 3.0,
 //       2.0, 5.0, 7.0,
 //       2.0, 5.0, 7.0};

 //    arr1.reshape({3,3});
 //    xarray<double> arr2
 //      {5.0, 6.0, 7.0};

 //    xarray<double> res = view(arr1, 1) + arr2;

 //    cout << res << std::endl;

    return 0;
}