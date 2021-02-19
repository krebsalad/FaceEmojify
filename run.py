import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime

# image class for conversion to mat
class Image:
    def __init__(self, _id, _pixels,_emotion=-1,_usage="Training"):
        self.id = _id

        # img data
        self.pixels = _pixels
        self.mat = None
        self.matCreated = False
        self.shape = None

        # train data
        self.emotion = _emotion
        self.usage = _usage
        
        # predicion data
        self.p_emotion = -1

    def __str__(self):
        return str(self.pixels)

    def __getitem__(self, key):
            return self.pixels[key]

    def getShape(self):
        if(self.shape == None):
            s = 0
            lenP = len(self.pixels)
            if(lenP%2 != 0):
                return None
            else:
                s = int(math.sqrt(lenP))
            if(s * s != lenP):
                s = 0
            self.shape = (s, s)
        return self.shape

    def setPixels(self, _pixels):
        self.pixels = _pixels
        self.mat = None
        self.matCreated = False
        self.shape = None

    def setFromCvMat(self, _mat):
        self.mat = _mat
        s = self.mat.shape
        self.matCreated = True
        self.shape = s
        self.pixels = [0] * (s[0] * s[1])
        for x in range(0, s[0]):
            for y in range(0, s[1]):
                self.pixels[(x * s[1]) + y] = self.mat[x, y]

    def getCvMat(self):
        if self.matCreated:
            return self.mat
        s = self.getShape()
        self.mat = np.zeros(s, np.uint8)
        for x in range(0, s[0]):
            for y in range(0, s[1]):
                self.mat[x, y] = self.pixels[(x * s[1]) + y]
        self.matCreated = True
        return self.mat


# read images from csv with headers emotion, Usage, pixels
def parseHeaders(headerList):
    dataType = ''
    for h in headerList:
        h2 = h.replace(' ', '')
        dataType += h2[0]
    return dataType.lower()

def readImagesFromCsv(path, max_n=0):
    print("reading image data ", path)

    # read data
    data_f = None
    if(os.path.isfile(path)):
        data_f = open(path)
    if (data_f == None):   
        print("no data found")
        return []
    data = data_f.readlines()
    data_f.close()

    # create image objects
    images = []
    dataType = ''
    for i, line in enumerate(data):
        line = line.replace('\n', "").replace('"', "").split(",")
        if(i == 0):
            dataType = parseHeaders(line)
            if(dataType == ''):
                print('no headers found')
                break
            print("found headers:", line, ", set as dataType:", dataType)
            continue

        img = None
        if (dataType == 'ep'):
            img = Image(i, np.asarray(line[1].split(" "), dtype=np.uint8, order='C'), int(line[0]),"Training")
            images.append(img)
            
        elif (dataType == 'p'):
            img = Image(i, np.asarray(line[0].split(" "), dtype=np.uint8, order='C'), _usage="PrivateTest")
            images.append(img)

        elif (dataType == 'eup'):
            img = Image(i, np.asarray(line[2].split(" "), dtype=np.uint8, order='C'), int(line[0]), _usage=line[1])
            images.append(img)
        else:
            print("no implementation for datatype:", dataType)
            return []

        if(max_n != 0 and i > max_n-1):
            break
    print("read data succefully: length of images ", len(images))
    return images

def getImagesAsDataLists(_images):
    pixelsList = []
    emotionsList = []
    usageList = []
    for im in _images:
        pixelsList.append(im.pixels)
        emotionsList.append(im.emotion)
        usageList.append(im.usage)
    return (pixelsList, emotionsList, usageList)

def getImagesAsCvDataLists(_images):
    imagesList = []
    emotionsList = []
    usageList = []
    for im in _images:
        imagesList.append(im.getCvMat())
        emotionsList.append(im.emotion)
        usageList.append(im.usage)
    return (imagesList, emotionsList, usageList)

# opencv util functions
def addEmotionToImage(_image, _emotion):
    emoji_mat = None
    img = _image.getCvMat()
    if _image.p_emotion > -1 and _image.p_emotion < 7:
        emoji_mat = cv2.imread("resources/emojis/" + str(_emotion) + ".PNG")
    else:
        emoji_mat = cv2.imread("resources/emojis/cross.PNG")

    emoji_mat = cv2.cvtColor(emoji_mat, cv2.COLOR_BGR2GRAY)
    emoji_mat = cv2.resize(emoji_mat, img.shape)
    return np.concatenate((img, emoji_mat), axis=1)

def showImages(_images, max_n=0, _showPredictedEmotion=False):
    print("showing " + len(_images) + " images, press <esc> in the opencv window to quit")
    for i, img in enumerate(_images):
        if(max_n != 0 and i > max_n-1):
            break

        if _showPredictedEmotion:
            cv2.imshow("prediction image", addEmotionToImage(img, img.p_emotion))
        else:
            cv2.imshow('image', img.getCvMat())

        key = cv2.waitKey(0)
        if key == ord('\x1b'):
            break

def writeImages(_images, max_n=0, _showPredictedEmotion=False):
    print("writing images...")
    for i, img in enumerate(_images):
        if(max_n != 0 and i > max_n-1):
            break

        if _showPredictedEmotion:
            cv2.imwrite("images/image_" + str(i) + ".jpg", addEmotionToImage(img, img.p_emotion))
        else:
            cv2.imwrite("images/image_" + str(i) + ".jpg", img.getCvMat())


def plotHistory(history,epochs,showPlot=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')


    plt.savefig('images/figure2_' + str(datetime.timestamp(datetime.now())) + '.png')
    if showPlot:
        plt.show()


# others
def calculateGradientImage(_image):
    im = np.float32(_image.getCvMat().copy())

    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return (gx, gy, mag, angle)

def splitListsForTraining(sampleList, targetList, randomize=True, dividor=4):
    # get random indecies
    sample_indecies = []
    if randomize:
        sample_indecies = random.sample(range(len(sampleList)), int(len(sampleList)/dividor))
    else:
        sample_indecies = range(int(len(sampleList)/dividor))

    validation_indecies = [i for i in range(len(sampleList)) if i not in sample_indecies]
    
    # set the random images
    sample_images = []
    validation_images = []
    for i in sample_indecies:
        sample_images.append(sampleList[i])
    for i in validation_indecies:
        validation_images.append(sampleList[i])

    # set the random targets
    sample_targets = []
    validation_targets = []
    for i in sample_indecies:
        sample_targets.append(targetList[i])
    for i in validation_indecies:
        validation_targets.append(targetList[i])

    return (sample_images, sample_targets, validation_images, validation_targets)

#classification
def getKNeighborsClassifier(pixelsList, emotionsList):
    from sklearn import neighbors
    # setup classifier
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    return trainClassifier(classifier, pixelsList, emotionsList)

def getRandomForestClassifier(pixelsList, emotionsList):
    from sklearn import ensemble
    # setup classifier
    classifier = ensemble.RandomForestClassifier()
    return trainClassifier(classifier, pixelsList, emotionsList)


def getCNNClassifier(matList, emotionsList, datasetDividor=5, epochs=500, image_shape=(48,48), modelSavePath='models/lastUsedModel.keras', loadModelPath=None, showPlot=False):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, BatchNormalization
    from keras.constraints import maxnorm
    from keras.optimizers import Adam

    model = None
    if loadModelPath:
        if os.path.isfile(loadModelPath):
            print("loading model", loadModelPath)
            model = tf.keras.models.load_model(loadModelPath)
        else:
            print("could not find model", loadModelPath)
            print("exiting...")
            sys.exit(1)
    else:
        loadModelPath = None

    if loadModelPath == None:
        # creating model 
        print("No model given, creating new model...")
        print("adding layers...")
        model = Sequential()
        model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(image_shape[0], image_shape[1], 1)))
        model.add(MaxPool2D())

        model.add(Conv2D(64,3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(128,3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(256,3, padding="same",activation="relu"))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dropout(0.2))

        model.add(Dense(128,activation="relu"))
        model.add(Dense(7, activation="softmax"))

        model.summary()

        # compiling model
        print("compiling model...")
        opt = Adam(lr=0.000001)
        model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

    # load onlt the model
    if epochs == 0 and loadModelPath != None:
        print("skipped training...")    
        return model

    # split data into sample and validation sets
    sample_images, sample_targets, validation_images, validation_targets = splitListsForTraining(matList, emotionsList, dividor=datasetDividor)

    sample_images = np.array(sample_images) / 255
    validation_images = np.array(validation_images) / 255

    sample_images = tf.reshape(sample_images, (-1, image_shape[0], image_shape[1], 1))
    sample_targets = np.array(sample_targets)

    validation_images = tf.reshape(validation_images, (-1, image_shape[0], image_shape[1], 1))
    validation_targets = np.array(validation_targets)

    # training
    print("Training with n of", len(sample_images))
    print("validating with n of", len(validation_images))

    history = model.fit(sample_images,sample_targets,epochs = epochs , validation_data = (validation_images, validation_targets))


    plotHistory(history,epochs,showPlot=showPlot)

    # save the model
    if os.path.isfile(modelSavePath):
        modelSavePath = modelSavePath.replace('.keras', '')
        modelSavePath += str(datetime.timestamp(datetime.now())) + '.keras'

    print("Saving model as", modelSavePath)
    model.save(modelSavePath)
    return model

def trainClassifier(classifier, pixelsList, emotionsList):
    # split data into sample and validation sets
    sample_images, sample_targets, validation_images, validation_targets = splitListsForTraining(pixelsList, emotionsList)

    # train
    print("Training with n of", len(sample_images))
    classifier.fit(sample_images, sample_targets)

    print("validating with n of", len(validation_images))
    score = classifier.score(validation_images, validation_targets)

    print("Classifier score:", str(score))

    return classifier

def setPredictions(_classifier, _images, usingCNN=False, image_shape=(48,48)):
    for image in _images:
        # predict 
        if not usingCNN:
            p = _classifier.predict([image.pixels])[0]
            print(image.usage, "img", image.id, "predicted:", p)
        else:
            temp = [image.getCvMat().copy()]
            temp = np.array(temp) / 255
            temp = tf.reshape(temp, (-1, image_shape[0], image_shape[1], 1))
            res = _classifier.predict(temp)[0]
            highestProbI = -1
            highestProb = 0
            for i, prob in enumerate(res):
                if prob > highestProb:
                    highestProbI = i
                    highestProb = res[i]
            p = highestProbI
            print(image.usage, "img", image.id, "predicted:", p, res)    
        
        image.p_emotion = p

def plotClasses(_images, _show=False, _classRange=7):
    classCount = [0] * _classRange
    for img in _images:
        if img.emotion != -1:             
            classCount[img.emotion] += 1
    print("class count ", classCount)
    plt.bar(range(len(classCount)), classCount)
    if(_show): 
        plt.show()
    plt.savefig('images/figure1_' + str(datetime.timestamp(datetime.now())) + '.png')
    return True


def solution1(train_images, test_images):
    # train
    pixelsList, emotionsList, usageList = getImagesAsDataLists(train_images)
    classifier = getRandomForestClassifier(pixelsList, emotionsList)

    # predict
    setPredictions(classifier, test_images)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)
    

def solution2(train_images, test_images):
    # train
    pixelsList, emotionsList, usageList = getImagesAsDataLists(train_images)
    classifier = getKNeighborsClassifier(pixelsList, emotionsList)

    # predict
    setPredictions(classifier, test_images)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

def solution3(train_images, test_images):
    # train
    matList, emotionsList, usageList = getImagesAsCvDataLists(train_images)
    classifier = getCNNClassifier(matList, emotionsList, datasetDividor=2,epochs=200)

    # predict
    setPredictions(classifier, test_images, usingCNN=True)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

# main prog
def main():
    # read data
    train_images = readImagesFromCsv("resources/train.csv")
    test_images = readImagesFromCsv("resources/test.csv", max_n=100)

    # pre analysing
    plotClasses(train_images)

    # solution1(train_images, test_images)
    # solution2(train_images, test_images)
    solution3(train_images, test_images)

    sys.exit(0)

if __name__ == "__main__":
    main()