from knn_classification import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax

# logistic regression will get this function :
# z = a x (weight) + b x (length) + c x (diagonal) + d x (height) + e x (width) + f
# sigmoid function : it changes z vaulue to be smaller than 1 and larger than 0. So, the value can indicate probability.
z = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z)) # sigmoid function
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()


# Using logistic regression, we will do binary classification first
# we will extract Bream and Smelt data, and ignore the rest of data just for now.
# And we're going to make logistic regression model to classify bream and smelt.

# Boolean indexing(to extract bream and smelt)
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True,False,True,False,False]]) # ['A' 'C']

# when you want to choose Bream and Smelt, do like this.
bream_smelt_indexes = (train_target=='Bream') | (train_target=='Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
print(lr.predict(train_bream_smelt[:5])) # ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
print(lr.predict_proba(train_bream_smelt[:5]))
# [[0.99759855 0.00240145] # [probability for bream, probability for smelt]
#  [0.02735183 0.97264817]
#  [0.99486072 0.00513928]
#  [0.98584202 0.01415798]
#  [0.99767269 0.00232731]]
print(lr.classes_) # ['Bream' 'Smelt']

# You might wonder why it is nor 100% or 0%. Because we gave the model that has been trained already.
# Answer : linear regression model (including logistic regression) makes equation that can show tendency of the circumstance "using" train_input.
# So, after it finishes to make model, it judges new input using the equation. Not all of the train_input. It doesn't compare with train_input.

# let's check coefficient of trained model.
print(lr.coef_,lr.intercept_) # [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# so, this model has made the equation like this :
# z = -0.4037798 x (weight) -0.57620209 x (length) -0.66280298 x (diagonal) -1.01290277 x (height) -0.73168947 x (width) -2.16155132

# Let's see some of z values
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions) # [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]

# let's input these to sigmoid funcrion. It will return data between 0 and 1.
print(expit(decisions)) # [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]



# Let's do multiclass classification using logistic regression

# Logistic regression uses repetitive algorithm.
# Data we have is quite small. So, we should repeat 1000 times for better trainig.
# And, LogisticRegression regulates square of coefficient like Ridge regression. But the difference is LogisticRegression has parameter C rather alpha.
# The larger the C, the weaker the regulation. Rather, the larger the alpha, the stronger the regulation in Ridge regression.

# Default value of C is 1. But we will set it as 20 to ease regulation.
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) # 0.9327731092436975
print(lr.score(test_scaled, test_target))   # 0.925
print(lr.predict(test_scaled[:5])) # ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
# [[0.    0.014 0.841 0.    0.136 0.007 0.003]
#  [0.    0.003 0.044 0.    0.007 0.946 0.   ]
#  [0.    0.    0.034 0.935 0.015 0.016 0.   ]
#  [0.011 0.034 0.306 0.007 0.567 0.    0.076]
#  [0.    0.    0.904 0.002 0.089 0.002 0.001]]

print(lr.classes_) # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

# there are 7 equations. So, there are 7 z values.
print(lr.coef_.shape,lr.intercept_.shape) #(7, 5) (7,)

# The softmax function makes the sum of 7 values ​​equal to 1.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
# [[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
#  [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
#  [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
#  [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
#  [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]
# these are z values. we will change this value using softmax.

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
# [[0.    0.014 0.841 0.    0.136 0.007 0.003]
#  [0.    0.003 0.044 0.    0.007 0.946 0.   ]
#  [0.    0.    0.034 0.935 0.015 0.016 0.   ]
#  [0.011 0.034 0.306 0.007 0.567 0.    0.076]
#  [0.    0.    0.904 0.002 0.089 0.002 0.001]]


