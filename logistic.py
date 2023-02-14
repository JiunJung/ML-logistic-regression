from knn_classification import *
import matplotlib.pyplot as plt

# logistic regression will get this function :
# z = a x (weight) + b x (length) + c x (diagonal) + d x (height) + e x (width) + f
# sigmoid function : it changes z vaulue to be smaller than 1 and larger than 0. So, the value can indicate probability.
z = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z)) # sigmoid function
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()





