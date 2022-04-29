import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

df = pd.read_csv('train.csv')

df['Sex'] = df['Sex'].replace(['male', 'female'], [0, 1])

df = df.drop(
    columns=
    ["PassengerId", 
    "Name", 
    "Age",
    "Ticket",
    "Cabin",
    "Embarked"
    ]
)
df = pd.get_dummies(df)

print(df.head())

from sklearn.model_selection import train_test_split
y = df.Survived
X = df.drop(columns = 'Survived')

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
print(X_train.shape)
print(y_train.shape)

X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1, 1)
print(X_test.shape)
print(y_test.shape)

np.save('titanic_X_train.npy', X_train)
np.save('titanic_y_train.npy', y_train)
np.save('titanic_X_test.npy', X_test)
np.save('titanic_y_test.npy', y_test)

"""
from nn_2layers import NeuralNet

neuralnet = NeuralNet(X_train.shape[1], 2, 1, lr=0.01, X_test=None, y_test=None)

w_1 = [[0.0, 0.0, 0.0, 0.0, 0.0], [5.853552954208808e-10, -1.2354092344902669e-09, -3.017959160808027e-10, -1.5164403066592058e-10, -1.7486119905001385e-12]]
b_1 = [0.0, -99.99999999768139]
w_2 = [1.855836391486696, -399709892149.4715]
b_2 = -20.695247345136565

K = torch.Tensor(w_1)
neuralnet.model.fc[0].weight = torch.nn.Parameter(K)
K = torch.Tensor(b_1)
neuralnet.model.fc[0].bias = torch.nn.Parameter(K)

K = torch.Tensor(w_2)
neuralnet.model.fc[1].weight = torch.nn.Parameter(K)
K = torch.Tensor(b_1)
neuralnet.model.fc[1].bias = torch.nn.Parameter(K)

#neuralnet.fit(X_train, y_train, num_iter=500)

print(neuralnet.score(X_train, y_train))

print(neuralnet.model.state_dict())
#OrderedDict([('fc.0.weight', tensor([[-0.3391,  0.4433,  0.1936,  0.0571, -0.0637],
#        [ 0.5108, -1.8536,  0.1705,  0.0323, -0.0020]], device='cuda:0')), 
# ('fc.0.bias', tensor([-0.2301,  1.1177], device='cuda:0')), 
# ('fc.2.weight', tensor([[ 0.0525, -1.3537]], device='cuda:0')), 
# ('fc.2.bias', tensor([1.6255], device='cuda:0'))])
"""