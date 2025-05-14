import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(0)
w1 = np.random.rand(2, 4)
w2 = np.random.rand(4, 1)

for _ in range(10000):
    l1 = sigmoid(X @ w1)
    l2 = sigmoid(l1 @ w2)
    d2 = (y - l2) * sigmoid_deriv(l2)
    d1 = (d2 @ w2.T) * sigmoid_deriv(l1)
    w2 += l1.T @ d2 * 0.1
    w1 += X.T @ d1 * 0.1

print(l2)
