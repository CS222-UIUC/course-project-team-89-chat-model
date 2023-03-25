#For now, this classifies images into categories
#yeah

#user specifies these parameters on the site
# TODO: fetch these values using API, input into file

data_file = 'spc0'
cols_to_drop = ['spc1']
to_pred = ['spc2']
opt = 'spc3'
max_acceptable_loss_1 = spc4
min_acc_2 = spc5
val_loss_3 = spc6
epoch_no_1 = spc7
epoch_no_2 = spc8
epoch_no_3 = spc8
test_image_classify = "spc9"



##################
#boilerplate code:

#import needed libraries
#data handling
import numpy as np 
import pandas as pd

#machine learning
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#plot results
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

#imports Pillow and IPython libraries to handle images and to demonstrate algorithm on our own faces
from PIL import Image
from IPython.display import Image as im

#get data (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)
#CSV contains age, gender, and ethnicity data, shortened to agdf
agdf = pd.read_csv(data_file)

#convert pixels into numpy array, printing head
agdf['pixels']=agdf['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

#drop unnecessary column "img_name"
agdf.drop(cols_to_drop, axis = 1)

#put images in numpy array to better handle data
x = np.array(agdf['pixels'].tolist())

#convert pixels from 1D to 3D
x = x.reshape(x.shape[0],48,48,1)

#model predicts gender, so use gender column
y = agdf[to_pred[0]]

#split data into 20% testing and 80% training
xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.20, random_state=42)

#create the convolutional neural network
genderModel = tf.keras.Sequential([
    L.InputLayer(input_shape=(48,48,1)), #input layer
    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #first convolutional layer
    L.BatchNormalization(), #normalize the batch
    L.MaxPooling2D((2, 2)), #
    L.Conv2D(64, (3, 3), activation='relu'), #another convolutional layer
    L.MaxPooling2D((2, 2)),
    L.Conv2D(64, (3, 3), activation='relu'), #another convolutional layer
    L.MaxPooling2D((2, 2)),
    L.Flatten(), #flatten layer to match dimensions
    L.Dense(64, activation='relu'),
    L.Dropout(rate=0.1), #changed dropout rate to 0.1, 0.5 caused it to not learn and stay at 0.4273 accuracy
    L.Dense(1, activation='sigmoid')
])

#optimize the model with sgd
genderModel.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<max_acceptable_loss_1):
            self.model.stop_training = True

callback = myCallback()

#prints the details of the model
genderModel.summary()

#train model on data with 20 epochs
history = genderModel.fit(xtr, ytr, epochs=epoch_no_1, validation_split=0.1, batch_size=64, callbacks=[callback])

acc = genderModel.evaluate(xt, yt, verbose = 1)
print("Score:", acc[0])
print("Accuracy:", acc[1])

#create the convolutional neural network
ethnicityModel = tf.keras.Sequential([
    L.InputLayer(input_shape=(48,48,1)), #input layer
    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #first convolutional layer
    L.MaxPooling2D((2, 2)), #
    L.Conv2D(64, (3, 3), activation='relu'), #another convolutional layer
    L.MaxPooling2D((2, 2)),
    L.Flatten(), #flatten layer to match dimensions
    L.Dense(64, activation='relu'),
    L.Dropout(rate=0.1), #changed dropout rate to 0.1, 0.5 caused it to not learn and stay at 0.4273 accuracy
    L.Dense(5)
])

#optimize the model with rmsprop
ethnicityModel.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#stop training when validation accuracy reaches 80%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>min_acc_2):
            self.ethnicityModel.stop_training = True
        
callback = myCallback()

#print model details
ethnicityModel.summary()

#train model
history = ethnicityModel.fit(xtr, ytr, epochs=epoch_no_2, validation_split=0.1, batch_size=64, callbacks=[callback])

#print the accuracy results
acc = ethnicityModel.evaluate(xt, yt, verbose = 1)
print("Score:", acc[0])
print("Accuracy:", acc[1])

#show the history of the loss
#I learned how to use plotly from https://plotly.com/python/
#I used plotly.express to clearly show the training history
fig = px.line(
    history.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'loss'}, 
    title='Training History')
fig.show()

#get predictions
genPred = genderModel.predict(xt)
genPred = genPred.flatten()

#output metrics
print(confusion_matrix(yt, genPred.round(0)))
print(classification_report(yt, genPred.round(0)))

#get ethnicity data, split data into training and testing
y = agdf['ethnicity']

xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.20, random_state=42)

#show the history of the loss
fig = px.line(
    history.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'loss'}, 
    title='Training History')
fig.show()

#get predictions
ethPred = ethnicityModel.predict(xt)

predictions = []

#convert prediction to categorical
for pr in ethPred:
    if max(pr) == pr[0]:
        predictions.append(0)
    elif max(pr) == pr[1]:
        predictions.append(1)
    elif max(pr) == pr[2]:
        predictions.append(2)
    elif max(pr) == pr[3]:
        predictions.append(3)
    else:
        predictions.append(4)

predictions = np.array(predictions)


#output metrics
print(confusion_matrix(yt, predictions))
print(classification_report(yt, predictions))

#get ethnicity data, split data into training and testing
y = agdf['age']

xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.20, random_state=42)

#create the convolutional neural network
ageModel = tf.keras.Sequential([
    L.InputLayer(input_shape=(48,48,1)), #input layer
    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #first convolutional layer
    L.BatchNormalization(), #normalize the batch
    L.MaxPooling2D((2, 2)), #
    L.Conv2D(64, (3, 3), activation='relu'), #another convolutional layer
    L.MaxPooling2D((2, 2)),
    L.Conv2D(64, (3, 3), activation='relu'), #another convolutional layer
    L.MaxPooling2D((2, 2)),
    L.Flatten(), #flatten layer to match dimensions
    L.Dense(64, activation='relu'),
    L.Dropout(rate=0.1), #changed dropout rate to 0.1, 0.5 caused it to not learn and stay at 0.4273 accuracy
    L.Dense(1, activation='relu')
])


#optimizers, compile
sgd = tf.keras.optimizers.SGD(momentum=0.9)
ageModel.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


#in order to prevent program from running too long, stop after val_loss is under 110
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<val_loss_3):
            print("\nModel reached sufficiently low loss (<110) so finish training early")
            self.model.stop_training = True
        
callback = myCallback()

#print model details
ethnicityModel.summary()

#train model
history = ageModel.fit(xtr, ytr, epochs=epoch_no_3, validation_split=0.1, batch_size=64, callbacks=[callback])

#print the error results
mse, mae = ageModel.evaluate(xt,yt,verbose=1)
print('Test Mean squared error: {}'.format(mse))
print('Test Mean absolute error: {}'.format(mae))

#show the history of the loss
fig = px.line(
    history.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'loss'}, 
    title='Training History')
fig.show()

#Imports Pillow Library to handle images, demonstrates algorithm on our own faces
from PIL import Image
from IPython.display import Image as im

#prediction function: prints prediction based on image
def predImage(file):
    image = Image.open(file)
    
    #converts image to 48x48 array of black and white pixels
    digit = np.asarray(image)
    digit = np.float32(digit)
    
    #displays image in the output
    fig = plt.figure(figsize=(5,5))
    plt.imshow(digit)
    plt.axis('off')
    
    #reshapes array to match shape needed for prediction
    digit = digit.reshape(1, 48, 48, 1)
    
    #displays gender prediction
    if genderModel.predict(digit).round() == 0:
        print("Predicted Gender: Male")
    else:
        print("Predicted Gender: Female")

    #displays ethnicity prediction
    if max(ethnicityModel.predict(digit)[0]) == ethnicityModel.predict(digit)[0][0]:
        print("Predicted Ethnicity: Caucasian")
    elif max(ethnicityModel.predict(digit)[0]) == ethnicityModel.predict(digit)[0][1]:
        print("Predicted Ethnicity: Asian")
    elif max(ethnicityModel.predict(digit)[0]) == ethnicityModel.predict(digit)[0][2]:
        print("Predicted Ethnicity: Latino")
    elif max(ethnicityModel.predict(digit)[0]) == ethnicityModel.predict(digit)[0][3]:
        print("Predicted Ethnicity: African")
    else:
        print("Predicted Ethnicity: Native American")
    
    #displays age prediction
    print("Predicted Age: " + str(int(ageModel.predict(digit)[0][0].round(0))))

#runs the above analysis on Sameer
predImage(test_image_classify)
