import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
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

# image specific util
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





