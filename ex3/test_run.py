
import numpy as np
import scipy as sp
import sys
from copy import copy
#from scipy.special import softmax

##@@@@@@@@@@@@@@@@@##
import pickle


# To see the output:
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


# neural network (single hidden layer)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def normalization_array(array):
  highest_num = 0
  for column in range(len(array[0])): #column
    for row in range(len(array)): #row
      array[row][column] = array[row][column]/255

def shuffle(train_x, y_result):
  assert len(train_x) == len(y_result)
  permutation = np.random.permutation(len(train_x))
  return train_x[permutation], y_result[permutation]
"""
def shuffle(trainx, trainy):
    np.random.shuffle(zip(trainx, trainy))
"""
def fprop(train_x, train_y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  #z2 = sigmoid(z2)
  #h2 = sp.special.softmax(z2)
  h2 = softmax(z2)
  z3 = 2# np.dot(y.T,h2) # 1X1 check @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  loss = 1# -np.log(z3)

  # result prints
  result = np.where(train_y == np.amax(train_y))
  #print ("z3=", z3, " loss=",loss, "res=",h2[result,0])
  #origin loss: loss = -(train_y * np.log(h2) + (1-train_y) * np.log(1-h2))
  ret = {'x': train_x, 'y': train_y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret



if __name__ == '__main__':


  # get W's and b's
  #####################
  filename = r"C:\israel train mnist\pythonProject1\0__res_hyper_atitsz.pkl"
  with open(filename, 'rb') as handle:
    bprop_cache = pickle.load(handle)


  W1 = bprop_cache['W1']
  W2 = bprop_cache['W2']
  b2 = bprop_cache['b2']
  b1 = bprop_cache['b1']
  param={'W1':W1, 'b1':b1, 'W2': W2, 'b2':b2}

  test_x = np.loadtxt("test_x")#, max_rows=rows)
  #$$$$$$


  #normalization:
  normalization_array(test_x)

  f=open("res_t.txt",'w')


  for index1 in range(len(test_x)):
    x_yguyt = np.zeros((1, 784))
    for koko in range(784):
        x_yguyt[0,koko] = test_x[index1,koko]
#####################

    x = x_yguyt.T
    y=0
    #one hot encoding:
    fprop_cache = fprop(x, y, param)
    y = fprop_cache['h2']
    temp_num=0
    temp_ind=0
    max_num= -10
    max_index= -1
    for indx in range(len(y)):
        if max_num < y[indx]:
           max_num = y[indx]
           max_index=indx
        
    #result = np.where(y == np.amax(y))
    print(max_index)

    f.write(str(max_index)+"\n")
  f.close()

   
