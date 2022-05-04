# Cost Function
# Cost (h(x), y) = - (y log(h(x)) + (1-y) log(1-h(x)))
# J = 1/m * sum (Cost(h(x), y))

def computeCost(X, y, theta):
  m = len(y)
  #print("theta.shape", theta.shape)
  #print("X.shape", X.shape)
  #print("y.shape", y.shape)

  Cost = - ((y * np.log(hypothesisFunc(X, theta))) + (1 - y) * np.log(1-hypothesisFunc(X, theta)))
  J = 1/m * np.sum(Cost, axis=0)

  grad = 1/m * np.sum((hypothesisFunc(X, theta) - y) * X, axis=0)
  
  return J, grad
