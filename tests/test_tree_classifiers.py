import unittest
from numbert.data_trans import Numbalizer,decode_vectorized
from numba.typed import List, Dict
import numpy as np



class TestDataTransfer(unittest.TestCase):
	def setUp(self):
		data1 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
		[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
		[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
		[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
		[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
		],np.bool);

		labels1 = np.asarray([3,1,1,1,2,2,2],np.int64);


		data2 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0,0,0,0,0], #1
		[0,0,1,0,0,0], #1
		[0,1,1,0,0,0], #1
		[1,1,1,0,0,1], #2
		[0,1,1,1,1,0], #2
		[1,1,1,0,1,0], #2
		],np.bool);

		labels2 = np.asarray([1,1,1,2,2,2],np.int64);
		data2 = data[:,[1,0,2,3,4,5]]


		data3 = np.asarray([
	#	 0 1 2 3 4 5 6 7 8 9 10111213141516
		[0,0], #1
		[1,0], #1
		[0,1], #1
		[1,1], #2
		],np.bool);

		labels3 = np.asarray([1,1,1,2],np.int64);