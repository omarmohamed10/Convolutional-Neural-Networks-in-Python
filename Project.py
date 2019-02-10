import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils , normalize
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten

imageData = []
imageLabels = []
faceDetectClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Import Pictures and detect Faces
def importPic(dir , foldername):
    DatasetPath = []
    for i in os.listdir(dir):
        DatasetPath.append(os.path.join(dir, i))

    counter = 0
    for i in DatasetPath:
        imgRead = cv2.imread(i)
        facePoints = faceDetectClassifier.detectMultiScale(imgRead, 1.3, 5)
        for (x, y, w, h) in facePoints:
            cropped = imgRead[y: y + h, x: x + w]
            resized_image = cv2.resize(cropped, (48, 48))
            name = foldername + "/pic" + str(counter)+".jpeg"
            cv2.imwrite(name, resized_image)
            counter += 1


def labelData(dir, label):
    DatasetPath = []
    for i in os.listdir(dir):
        DatasetPath.append(os.path.join(dir, i))

    for i in DatasetPath:
        imgRead = cv2.imread(i)
        grayImg = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
        imageData.append(grayImg)
        imageLabels.append(label)


importPic("D:\ml photos\Omar","Omar")
importPic("D:\ml photos\Ali","Ali")
importPic("D:\ml photos\Abdallah","Abdallah")
importPic("D:\ml photos\Magdy","Magdy")
importPic("D:\ml photos\Ghonem","Ghonem")
importPic("D:\ml photos\Ghareeb","Ghareeb")


labelData("D:\ml photos\Omar",0)
labelData("D:\ml photos\Ali",1)
labelData("D:\ml photos\Abdallah",2)
labelData("D:\ml photos\Magdy",3)
labelData("D:\ml photos\Ghonem",4)
labelData("D:\ml photos\Ghareeb",5)

imgData = np.array(imageData)

# Split Dataset into train & test
X_train , X_test , y_train , y_test = train_test_split(imgData , imageLabels , train_size=0.8 , random_state=20)

# number of classes = 6 for every person

nb_classes = 6
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''
def reShape2D(X_train,X_test):
    X_train = X_train.reshape(len(X_train), 48 , 48, 1)
    X_test = X_test.reshape(len(X_test), 48 , 48 , 1)
    return featureScale(X_train,X_test)

def featureScale(X_train , X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 225
    X_test /= 225
    return X_train,X_test
'''
'''
# Neural Networl using Keras
def NN(X_train, X_test , Y_train , Y_test , nb_classes):
    # scale image between 0...1
    X_train = normalize(X_train,axis=1)
    X_test = normalize(X_test, axis=1)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes , activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=10)

    loss, accuracy = model.evaluate(X_test,Y_test)
    print("Loss : " , loss)
    print("Accuracy : " , accuracy)
    model.save('nn.h5')

    #prediction = model.predict([X_test])
    #print(np.argmax(prediction[30]))


def CNN(X_train, X_test , Y_train , Y_test , nb_classes):
    X_train , X_test = reShape2D(X_train,X_test)

    model = Sequential()
    model.add(Conv2D(64,kernel_size=3 , input_shape=(48,48,1) ,activation='relu'))
    model.add(Conv2D(32,kernel_size=3 , activation='relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=20,epochs=3)

    loss , accuracy = model.evaluate(X_test,Y_test)
    print("Loss : ", loss)
    print("Accuracy : ", accuracy)
    model.save('cnn.h5')

    #prediction = model.predict([X_test])
    #print(np.argmax(prediction[30]))

'''

#predicting test[30] category
#plt.imshow(X_test[30])
#plt.show()


#NN(X_train,X_test,Y_train,Y_test,nb_classes)

#CNN(X_train,X_test,Y_train,Y_test,nb_classes)

