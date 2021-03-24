import cv2
import numpy as np
import math
import os
from datetime import datetime
import random


# python util
log_file_path = 'log.txt'
logFile = open(log_file_path, "w+")
logFile.close()

def writeToFile(file_path, txt):
    with open(file_path, "a") as myFile:
        myFile.write(txt + '\n')

def printLog(log_txt):
    print(log_txt)
    writeToFile(log_file_path, log_txt)

def release_list(a):
   del a[:]
   del a

def listFind(l, i):
    for i2 in l:
        if i == i2:
            return True
    return False 

def getModelSummaryAsString(model):
    str_list = []
    model.summary(print_fn=lambda c: str_list.append(c))
    model_summary_str = "\n".join(str_list)
    return model_summary_str

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
def getImagesEmotionsLists(_images):
    emotionsList = []
    for im in _images:
        emotionsList.append(im.emotion)
    return emotionsList

def getImagesAsDataLists(_images, dimensions=0):
    pixelsList = []
    emotionsList = []
    usageList = []
    for im in _images:
        if not dimensions:
            pixelsList.append(im.pixels)
        else:
            pixelsList.append(im.pixels[:dimensions])
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

def rotateCvMat(mat, angel):#parameter angel in degrees
    h = mat.shape[0]
    w = mat.shape[1]
    mat_rot = cv2.getRotationMatrix2D((w/2, h/2),angel, 1)
    mat_res = cv2.warpAffine(mat, mat_rot, (w, h), flags=cv2.INTER_LINEAR)
    return mat_res

# read images from csv with headers emotion, Usage, pixels
def parseHeaders(headerList):
    dataType = ''
    for h in headerList:
        h2 = h.replace(' ', '')
        dataType += h2[0]
    return dataType.lower()

def normalizeDataSet(images, max_n=0, n_emotions=7, max_duplications=3, augment_duplications=False):
    normalized_images = []

    # count amount of images per emotion
    feature_counts = [0] * n_emotions
    for img in images:
        if img.emotion == -1:
            continue
        feature_counts[img.emotion] += 1

    # calculate average and or set to max_sample size
    average_count = sum(feature_counts) / len(feature_counts)
    if(max_n != 0 and average_count > max_n/n_emotions):
        average_count = max_n/n_emotions
    
    # add images to normalized list
    normalized_feature_counts = [0] * n_emotions
    for img in images:
        if img.emotion == -1:
            continue

        if normalized_feature_counts[img.emotion] < average_count:
            normalized_images.append(img)
            normalized_feature_counts[img.emotion] += 1

    # fill in missing data
    random.seed(10)
    for i, count in enumerate(normalized_feature_counts):
        if count >= average_count:
            continue
        
        count_left = average_count - count
        duplications = 0
        while count_left > 0 and duplications < max_duplications:
            for img in images:
                if count_left <= 0:
                    break
                if img.emotion != i:
                    continue

                if duplications > 0 and augment_duplications:
                    r = random.randint(0, 1)
                    if r == 1:
                        img.setFromCvMat(cv2.flip(img.getCvMat(), 1))
                    r = random.randint(-5, 5)
                    img.setFromCvMat(rotateCvMat(img.getCvMat(), r))

                normalized_images.append(img)
                count_left -= 1
            duplications += 1

    return normalized_images


def readImagesFromCsv(path, max_n=0, usage_skip_list=[], normalize_data_set=False, normalize_augmentation=True):
    printLog("reading image data " + path)

    # read data
    data_f = None
    if(os.path.isfile(path)):
        data_f = open(path)
    if (data_f == None):   
        printLog("no data found")
        return []
    data_f.close()

    # create image objects
    images = []
    dataType = ''
    with open(path) as data_f:
        for i, line in enumerate(data_f):
            line = line.replace('\n', "").replace('"', "").split(",")
            if(i == 0):
                dataType = parseHeaders(line)
                if(dataType == ''):
                    printLog('no headers found')
                    break
                printLog("found headers: " + str(line) + ", set as dataType: " + dataType)
                continue
            
            img = None
            if (dataType == 'ep'):
                img = Image(i, np.asarray(line[1].split(" "), dtype=np.uint8, order='C'), int(line[0]),"Training")
                images.append(img)
            
            elif (dataType == 'p'):
                img = Image(i, np.asarray(line[0].split(" "), dtype=np.uint8, order='C'), _usage="PrivateTest")
                images.append(img)

            elif (dataType == 'eup'):
                if len(usage_skip_list):
                    if listFind(usage_skip_list, line[1]):
                        continue
                img = Image(i, np.asarray(line[2].split(" "), dtype=np.uint8, order='C'), int(line[0]), _usage=line[1])
                images.append(img)
            elif (dataType.find('eupp') != -1):
                if len(usage_skip_list):
                    if listFind(usage_skip_list, line[1]):
                        continue
                    
                pixels = line[2:len(line)]
                img = Image(i, np.asarray(pixels, dtype=np.float32, order='C'), int(line[0]), _usage=line[1])
                images.append(img)
            else:
                printLog("no implementation for datatype: " + dataType)
                return []

            if(max_n != 0 and i > max_n-1 and not normalize_data_set):
                break
    printLog("read data succefully: length of images " + str(len(images)))

    if normalize_data_set:
        images = normalizeDataSet(images, max_n=max_n, n_emotions=7, max_duplications=5, augment_duplications=normalize_augmentation)
    return images

def writeImagesAsCsv(path, images):
    print('writing images to csv file...')

    # clean contents
    out_csv_file = open(path, "w+")
    out_csv_file.close()

    with open(path, 'a') as out_csv_file:
        txt = 'emotion, Usage, pixels\n'
        out_csv_file.write(txt)
        for img in images:
            txt = str(img.emotion) + ','
            txt += img.usage + ','
            for p in img.pixels:
                txt += str(p) + ' '
            txt += '\n'
            out_csv_file.write(txt)
    
    print('wrote '+str(len(images))+' image succesfully')

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
    printLog("showing " + str(len(_images)) + " images, press <esc> in the opencv window to quit")
    for i, img in enumerate(_images):
        if(max_n != 0 and i > max_n-1):
            break

        if _showPredictedEmotion:
            cv2.imshow("prediction image", addEmotionToImage(img, img.p_emotion))
        else:
            print("emotion:", img.emotion)
            cv2.imshow('image', img.getCvMat())

        key = cv2.waitKey(0)
        if key == ord('\x1b'):
            break

def writeImages(_images, max_n=0, _showPredictedEmotion=False, name='model1'):
    printLog("writing images...")
    for i, img in enumerate(_images):
        if(max_n != 0 and i > max_n-1):
            break

        if _showPredictedEmotion:
            cv2.imwrite("images/"+name+"/image_" + str(i) + ".jpg", addEmotionToImage(img, img.p_emotion))
        else:
            cv2.imwrite("images/"+name+"/image_" + str(i) + ".jpg", img.getCvMat())





