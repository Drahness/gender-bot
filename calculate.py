

import cv2
import numpy as np
import pandas as pd
import random
import os
import gc
from keras.models import model_from_json
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img



def detect():
    print("detect")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    rows = 150
    columns = 150

    def process(imglist):
        X = []
        y = []
        for image in imglist:
            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (rows, columns),interpolation=cv2.INTER_CUBIC))
        return X, y


    images2 = ["image.png"]
    def testimage(imagec):
        X_test, y_test = process(imagec[0:1])
        x = np.array(X_test)
        print(x)
        test_datagen = ImageDataGenerator(rescale=1./255)
        for batch in test_datagen.flow(x, batch_size=1):
            pred = loaded_model.predict(batch)
            print(pred)
            pred = round(float(pred), 2)
            if pred > 0.5:
                return "Female, {}%".format(float(pred*100))
            else:
                return "Male, {}%".format(float(100-(pred*100)))
            

    return(testimage(images2))
