# Building Logistic Regression from Scratch in Python
print("Building Logistic Regression from Scratch in Python")

# ======================= Step 1: Importing Libraries ==========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================= Step 2: Loading the Data =============================
print("\nLoading the Dataset...")

dataset = np.loadtxt('/content/drive/MyDrive/[Coursera] Machine Learning By Andrew Ng/Week3-assignment/Logistic Regression/ex2data1.txt', delimiter=',')
#print(dataset[:5, :])

X = dataset[:, :-1]
y = dataset[:, -1]

y = y.reshape(len(y), 1)

# Print out some data points
print("First 5 Examples from the dataset: ")
#print(" x = \n{0} \n y = {1}".format(X[:5, :], y[:5]))
print("x = \n", X[:5, :])
print("y = ", y[:5])

m = len(y)
print("\nTraining Examples: ", m)
n = len(X[1])
print("Features: ", n)

# Add intercept term to X
print("\n\nAdding intercept term to X")
x_0 = np.ones((m,1))
X = np.concatenate((x_0, X), axis=1)
print("x = \n", X[:5, :])


# ======================= Step 3: Computing Initial Cost =======================
# Initiate theta and compute initial Cost
theta = np.zeros((n+1, 1))
print("\n\nInitializing theta = \n", theta)

print("\n\nComputing Initial Cost...")
# Save the Cost J in every iteration
cost, grad = computeCost(X, y, theta)

print("Initial Cost: ", cost[0])
print("Initial Grad: ", grad)


# Compute and display cost and gradient with non-zero theta
test_theta = [-24, 0.2, 0.2]
test_theta = np.array(test_theta).reshape((len(test_theta), 1))
cost, grad = computeCost(X, y, test_theta)

print("\nCost at test theta: ", cost[0])
print("Grad at test theta: ", grad)


# ======================= Step 4: Gradient Descent ============================
print("\n\nRunning Gradient Descent...")

# Choose some value of alpha
alpha = 0.01
num_iters = 400

theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Display Gradient Descent's results
print("\nTheta computed from gradient descent: \n", theta)

# ======================= Step 5: Plot the Convergence =========================
print("\n\n Convergence of Gradient Descent")
plt.plot(list(range(0,400)), J_history[0:400])
plt.xlabel("number of iterations")
plt.ylabel("Cost")
plt.show()

# ======================= Step 6: Predict ======================================
# Estimate the admission of a student with scores 45 and 85
X_test = np.array([1, 45, 85]).reshape(1,n+1)
price = np.matmul(X_test, theta)
print("\n\nPredicted admission of a student with scores 45 and 85 (using gradient descent): ", price[0][0])


# ======================= Step 7: Compute Accuracy =============================
p = np.matmul(X, theta)
print("\n\nTrain Accuracy: ", np.mean(((p>0.5)==y)) * 100)
