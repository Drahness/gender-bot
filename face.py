import cv2
import numpy as np
import pandas as pd
import random
import os
import gc
from keras.models import model_from_json
from PIL import Image







def don():
    train = "train/crop_part1/"
    imgs = "images/"

    train_male = ["male images directory"]
    train_female = ["female images directory"]
    trainimgs = train_male[:4000] + train_female[:4000]


    del train_male
    del train_female
    gc.collect()


    rows = 150
    columns = 150
    ch = 3

    def process(imglist):
        X = []
        y = []
        e = []
        a = []
        for image in imglist:
            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (rows, columns),interpolation=cv2.INTER_CUBIC))
            image = image.split("crop_part1/")[1]
            age = image.split("_")[0]
            a.append(age)
            image = image.split("_")
            print(image)
            if '0' in image[1]:
                print(image, "m")
                y.append(0)
            elif '1' in image[1]:
                print(image, "f")
                y.append(1)
        return X, y, a

    X, y, a = process(trainimgs)

    del trainimgs
    gc.collect()

    X = np.array(X)
    y = np.array(y)
    a = np.array(a)



    import sklearn
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

    del X
    del y
    gc.collect()

    ntrain = len(X_train)
    nval = len(X_val)

    batch_size=3

    from keras import layers
    from keras import models
    from keras import optimizers
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import img_to_array, load_img


   #  to continue training:
   # json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    #  load weights into new model
    #loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")

    #  evaluate loaded model on test data
    #loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator=val_datagen.flow(X_val, y_val, batch_size=batch_size)

    history = model.fit_generator(train_generator, steps_per_epoch=ntrain//batch_size, epochs=5, validation_data=val_generator, validation_steps=nval // batch_size)
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")



    
    print("Training complete")
        
don()


