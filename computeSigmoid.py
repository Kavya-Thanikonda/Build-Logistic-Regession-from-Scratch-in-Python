# Sigmoid Function
# 1 / (1 + e ^ (-z))

def sigmoidFunc(z):
  #print("theta.shape", theta.shape)
  #print("X.shape", X.shape)
  
  sigmoid = 1 / (1 + np.exp(-z))
  return sigmoid
