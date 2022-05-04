# Gradient Descent
# Repeat {
#   delta = 1/m * sum(h(x) - y) * x
#   theta = theta - alpha * delta  
# }
def gradientDescent(X, y, theta, alpha, num_iters):
  m = len(y)

  J_history = np.zeros((num_iters, 1))

  for i in range(num_iters):
    delta = 1/m * np.sum((hypothesisFunc(X, theta) - y) * X, axis=0)
    delta = delta.reshape(len(delta), 1)

    theta = theta - alpha * delta
    
    # Save the Cost J in every iteration
    J_history[i], grad = computeCost(X, y, theta)

  return theta, J_history
