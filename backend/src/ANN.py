# model parameters

fname = "banknote.csv"
to_pred = 'class'
dr = 0.1
lr = 0.001
neurons = [12, 6, 1]
epochs = 20
batch = 4

import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# preprocessing
bdf = pd.read_csv(fname)
x = bdf.drop([to_pred], axis = 1)
y = bdf[[to_pred]].values
xtr, xt, ytr, yt = train_test_split(x, y, test_size = 0.20, random_state = 42)

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xt = sc.transform(xt)

def create_model(LR, DR, ls_):
    #sequential model
    model = Sequential()
    #add dense layers
    model.add(Dense(ls_[0], input_dim = xtr.shape[1], activation = 'relu'))
    model.add(Dropout(DR))
    for i in range(1, len(ls_)):
	model.add(Dense(ls_[i], activation = 'relu'))
	model.add(Dropout(DR))
    model.add(Dense(1, activation = 'sigmoid'))
    #compiling:
    adam = Adam(lr = LR)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model(lr, dr, neurons[:-1])
modHist = model.fit(xtr, ytr, batch_size = batch, epochs = epochs, validation_split = 0.2, verbose = 0)
print("Accuracy:", model.evaluate(xt, yt, verbose = 1)[1])
