import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]

# observations for training
dx_train = d_X[:-20]
dy_train = d.target[:-20]
# observations for testing
dx_test = d_X[-20:]
dy_test = d.target[-20:]


def gradient_intercept(x, y):
    # function to calculate gradient and intercept, returns tuple with both values
    m = (np.mean(x) * np.mean(y) - np.mean(x * y)) / ((np.mean(x))**2 - np.mean(x ** 2))
    b = np.mean(y) - m * np.mean(x)
    return m, b


result = gradient_intercept(dx_train.squeeze(), dy_train.squeeze())

# gradient and intercept
m = result[0]
b = result[1]

# plotting graphs with label for the legend
plt.scatter(dx_train, dy_train, c='red', label='train')
plt.scatter(dx_test, dy_test, c='green', label='test')
# line of best-fit for training data
plt.plot(dx_train, m*dx_train + b, c='blue', label='best-fit')
plt.legend()
plt.show()
