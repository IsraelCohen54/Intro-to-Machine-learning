import numpy as np
import scipy as sp
import sys
from scipy.special import softmax

##@@@@@@@@@@@@@@@@@##
import pickle


# To see the output:
import matplotlib.pyplot as plt

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
def fprop_const(train_x, train_y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  #z2 = sigmoid(z2)
  h2 = sp.special.softmax(z2)
  z3 = np.dot(y.T,h2) # 1X1 check @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def fprop(train_x, train_y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  #z2 = sigmoid(z2)
  h2 = sp.special.softmax(z2)
  z3 = np.dot(y.T,h2) # 1X1 check @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  loss = -np.log(z3)

  # result prints
  result = np.where(train_y == np.amax(train_y))
  #print ("z3=", z3, " loss=",loss, "res=",h2[result,0])
  #origin loss: loss = -(train_y * np.log(h2) + (1-train_y) * np.log(1-h2))
  ret = {'x': train_x, 'y': train_y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
  dz2 = (h2 - y)  # dL/dz2
  dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
  db2 = dz2  # dL/dz2 * dz2/db2
  dz1 = np.dot(fprop_cache['W2'].T,
               (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
  dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

if __name__ == '__main__': 

  """
  I used pickle, its in """""" down below because path is other than your comp
  """
  train_x_data, train_y_data, test_x_data = sys.argv[1], sys.argv[2], sys.argv[3]

  train_x=np.loadtxt(train_x_data)
  train_y=np.loadtxt(train_y_data)
  test_x=np.loadtxt(test_x_data)

  """
  rows = 500
  train_x = np.loadtxt("train_x", max_rows=rows)
  train_y = np.loadtxt("train_y", max_rows=rows, dtype=int)
  test_x = np.loadtxt("test_x", max_rows=rows)

  W1 = np.random.rand(784, 784) * 0.1
  b1 = np.random.rand(784, 1) * 0.1
  W2 = np.random.rand(10, 784) * 0.1
  b2 = np.random.rand(10,1) * 0.1
  """
  W1 = np.random.randn(128, 784)
  b1 = np.random.rand(128,1)
  W2 = np.random.randn(10, 128)
  b2 = np.random.rand(10,1)
  b2 = np.zeros_like(b2)
  b1 = np.zeros_like(b1)
  params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  #normalization:
  normalization_array(train_x)
  normalization_array(test_x)

  #shuffle:
  train_x, train_y = shuffle(train_x,train_y)

  #creating vec to one hot encoding: y=[10]
  #y = np.zeros((1, 10))
  y = np.zeros((10,1))
  x = np.zeros(len(train_x[0]))
  # add while loop with 10 epoch (or more to check) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  epoch = 10
  while (epoch != 0):

    for index1 in range(len(train_x)):
      #####################
      x_yguyt = np.zeros((1, 784))
      for koko in range(784):
        x_yguyt[0,koko] = train_x[index1,koko]

      x = x_yguyt.T
      y_temp = int(train_y[index1])
      #one hot encoding:
      y[y_temp] = 1
      fprop_cache = fprop(x, y, params)
      bprop_cache = bprop(fprop_cache)
      y[y_temp] = 0

      #####################
      """  
    #x1 = np.zeros((1, 784))
    for (row_index, y_ind) in zip(train_x,train_y):
      np.copyto(row_index, x)
      y[y_ind] = 1
      fprop_cache = fprop(x, y, params)
      bprop_cache = bprop(fprop_cache)
      y[y_ind] = 0
      """
      #print(params.keys())
      eps = 0.01

      fprop_cache['W1'] -= bprop_cache['W1']*eps
      fprop_cache['W2'] -= bprop_cache['W2']*eps
      fprop_cache['b1'] -= bprop_cache['b1']*eps
      fprop_cache['b2'] -= bprop_cache['b2']*eps

      W1 -= eps*bprop_cache['W1']
      W2 -= eps * bprop_cache['W2']
      b2 -= eps * bprop_cache['b2']
      b1 -= eps * bprop_cache['b1']
      param = {'W1':W1, 'b1':b1, 'W2': W2, 'b2':b2}

    """  epoch -= 1
      eps = eps * 0.9
    outfile = open(r"C:\\Users\\Israel\\PycharmProjects\\pythonProject1\\res_hyper_atitsz.txt",'wb')
    hyper_params_dict = {'W1':fprop_cache['W1'],'W2':fprop_cache['W2'],'b2':fprop_cache['b2'],'b1':fprop_cache['b1']}
    pickle.dump(hyper_params_dict,outfile)
    outfile.close()
    """
    epoch -= 1
    eps = eps * 0.9
    print("epoch num: ", 10 - epoch)
    """
    filename = r"C:\\israel train mnist\\pythonProject1\\"+str(epoch)+"__res_hyper_atitsz.pkl"
    
    outfile = open(filename,'wb')
    hyper_params_dict = {'W1':fprop_cache['W1'],'W2':fprop_cache['W2'],'b2':fprop_cache['b2'],'b1':fprop_cache['b1']}
    pickle.dump(hyper_params_dict,outfile)
    
    outfile.close()
    """
  #for len(test_x))
  """
  # Compare numerical gradients to those computed using backpropagation algorithm
    for key in params:
        print(key)
        # These should be the same
        #print(bprop_cache[key])
        #  print(ng_cache[key])
  """
