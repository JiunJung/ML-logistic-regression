import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

fish = pd.read_csv("./fish_csv_data.csv")

print(fish.head())
#   Species  Weight  Length  Diagonal   Height   Width
# 0   Bream   242.0    25.4      30.0  11.5200  4.0200
# 1   Bream   290.0    26.3      31.2  12.4800  4.3056
# 2   Bream   340.0    26.5      31.1  12.3778  4.6961
# 3   Bream   363.0    29.0      33.5  12.7300  4.4555
# 4   Bream   430.0    29.0      34.0  12.4440  5.1340

print(pd.unique(fish['Species']))
#['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

# "Species" will be the target. So, we will make the other categories to input data.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy() # If you give list of coulmn categories, those coulmn data will be selected.
print(fish_input[:5])
# [[242.      25.4     30.      11.52     4.02  ]
#  [290.      26.3     31.2     12.48     4.3056]
#  [340.      26.5     31.1     12.3778   4.6961]
#  [363.      29.      33.5     12.73     4.4555]
#  [430.      29.      34.      12.444    5.134 ]]

# Make target data for training the model.
fish_target = fish['Species'].to_numpy()

# shuffle and split into train and test data
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# first, we will see what happens when we use KNN classification.
# we need to preprocess the data first. we use StandardScaler.
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# use KNN classification
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target)) #0.8907563025210085
print(kn.score(test_scaled, test_target))   #0.85

print(kn.classes_) # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish'] # arranged by alphabet number
print(kn.predict(test_scaled[:5])) # ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
# [[0.     0.     1.     0.     0.     0.     0.    ] # probability for each species
#  [0.     0.     0.     0.     0.     1.     0.    ]
#  [0.     0.     0.     1.     0.     0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
# [['Roach' 'Perch' 'Perch']] #this is why 4th input is decided as perch

#We use only near 3 data so there are only 0/3, 1/3, 2/3, 3/3 for probability.
#It's time to use logistic regression.





