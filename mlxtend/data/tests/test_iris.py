from mlxtend.data import iris_data
from mlxtend.data import wine_data
from mlxtend.data import autompg_data
from mlxtend.data import mnist_data
from mlxtend.data import boston_housing_data
from mlxtend.data import three_blobs_data
import numpy as np
import os
import re


this_dir, this_filename = os.path.split(__file__)
this_dir = re.sub("^stset","",this_dir[::-1])[::-1]
DATA_PATH = os.path.join(this_dir, "data", "iris.csv.gz")




def test_iris_data_uci():
	tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
	original_uci_data_x, original_uci_data_y = tmp[:, :-1], tmp[:, -1]
	original_uci_data_y = original_uci_data_y.astype(int)
	assert(original_uci_data_x,original_uci_data_y == iris_data()) 

def test_iris_data_r():
	tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
	original_r_data_x, original_r_data_y = tmp[:, :-1], tmp[:, -1]
	original_r_data_y = original_r_data_y.astype(int)
	original_r_data_x[34] = [4.9, 3.1, 1.5, 0.2]
	original_r_data_x[37] = [4.9, 3.6, 1.4, 0.1]
	assert(original_r_data_x,original_r_data_x == iris_data(version='r'))	