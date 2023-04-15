#user specifies these parameters on the site
data_file = 'spc0'
cols_to_drop = ['spc1']
to_pred = 'spc2'
opt = 'spc3'
max_acceptable_loss = spc4
epoch_no_1 = spc5
test_image_classify = "spc6"
imgx = spc7
imgy = spc8
conv_layers = spc9 #3



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
df = pd.read_csv(data_file)

#convert pixels into numpy array, printing head
df['pixels']=df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

#drop unnecessary column "img_name"
df.drop(cols_to_drop, axis = 1)

#put images in numpy array to better handle data
x = np.array(df['pixels'].tolist())

#convert pixels from 1D to 3D
x = x.reshape(x.shape[0],imgx,imgy,1)

#model predicts gender, so use gender column
y = df[to_pred]

#split data into 20% testing and 80% training
xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.20, random_state=42)

layer_list = [L.InputLayer(input_shape=(imgx,imgy,1))]
for i in range(conv_layers):
    if i == 0:
        layer_list.append(L.Conv2D(int((imgx + imgy)/3), (3, 3), activation='relu', input_shape=(int(imgx*2/3), int(imgy*2/3), 3)))
        layer_list.append(L.BatchNormalization())
        layer_list.append(L.MaxPooling2D((2, 2)))
    else:
        layer_list.append(L.Conv2D(int((imgx + imgy)*2/3), (3, 3), activation='relu'))
        layer_list.append(L.MaxPooling2D((2, 2)))
layer_list.append(L.Flatten())
layer_list.append(L.Dense(int((imgx + imgy)*2/3), activation='relu'))
layer_list.append(L.Dropout(rate=0.1))
layer_list.append(L.Dense(1, activation='sigmoid'))

#create the convolutional neural network
CNN_model = tf.keras.Sequential(layer_list)

#optimize the model with sgd
CNN_model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('val_loss')<max_acceptable_loss):
            self.model.stop_training = True

callback = myCallback()

#prints the details of the model
CNN_model.summary()

#train model on data with 20 epochs
history = CNN_model.fit(xtr, ytr, epochs=epoch_no_1, validation_split=0.1, batch_size=int((imgx + imgy)*2/3), callbacks=[callback])

acc = CNN_model.evaluate(xt, yt, verbose = 1)
print("Score:", acc[0])
print("Accuracy:", acc[1])

#prediction function: prints prediction based on image
def predImage(file):
    image = Image.open(file)
    
    #converts image to array of black and white pixels
    digit = np.asarray(image)
    digit = np.float32(digit)
    
    #displays image in the output
    fig = plt.figure(figsize=(5,5))
    plt.imshow(digit)
    plt.axis('off')
    
    #reshapes array to match shape needed for prediction
    digit = digit.reshape(1, imgx, imgy, 1)
    
    #displays prediction
    print("Predicted "" Value: " + str(CNN_model.predict(digit)))

#runs the above analysis on Sameer
predImage(test_image_classify)
