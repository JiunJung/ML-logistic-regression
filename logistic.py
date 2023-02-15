from knn_classification import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

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



