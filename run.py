from FE_image_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf

# for general use
def splitInstancesForTraining(train_instances, randomize=True, dividor=4):
    # get random indecies
    sample_indecies = []
    if randomize:
        import random
        sample_indecies = random.sample(range(len(train_instances)), int(len(train_instances)/dividor))
    else:
        sample_indecies = range(int(len(train_instances)/dividor))

    validation_indecies = [i for i in range(len(train_instances)) if i not in sample_indecies]
    
    # set the random images
    sample_instances = []
    validation_instances = []
    for i in sample_indecies:
        sample_instances.append(train_instances[i])
    for i in validation_indecies:
        validation_instances.append(train_instances[i])

    return (sample_instances, validation_instances)

# classifier
def trainClassifier(classifier, train_instances):
    # split data into sample and validation sets
    sample_instances, validation_instances = splitInstancesForTraining(train_instances)

    sample_features, sample_targets, sample_usage = getImagesAsDataLists(sample_instances)
    validation_features, validation_targets, validation_usage = getImagesAsDataLists(validation_instances)
    
    # train
    print("Training with n of", len(sample_features))
    classifier.fit(sample_features, sample_targets)

    print("validating with n of", len(validation_features))
    score = classifier.score(validation_features, validation_targets)

    print("Classifier score:", str(score))

    return classifier

def getKNeighborsClassifier(train_instances):
    from sklearn import neighbors
    # setup classifier
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    return trainClassifier(classifier, train_instances)

def getRandomForestClassifier(train_instances):
    from sklearn import ensemble
    # setup classifier
    classifier = ensemble.RandomForestClassifier()
    return trainClassifier(classifier, train_instances)


def visualizeModelFitlers(model, _show=False):
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        if (_show):
            plt.imshow(filters[:,:,:,0], cmap='gray')
        break # TODO!!!

# NN
def getCNNClassifier(train_images, datasetDividor=5, epochs=500, image_shape=(48,48), modelSavePath='models/lastUsedModel.keras', loadModelPath=None, showPlot=False, useTensorBoard=False):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, BatchNormalization
    from keras.constraints import max_norm
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
        model.add(Conv2D(32,(3,3),padding="same", activation="relu", input_shape=(image_shape[0], image_shape[1], 1)))
        model.add(MaxPool2D())

        model.add(Conv2D(64,(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(128,(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(256,(3,3), padding="same",activation="relu", kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dropout(0.2))

        model.add(Dense(128,activation="relu"))
        model.add(Dense(7, activation="softmax"))

        model.summary()
        visualizeModelFitlers(model)

        # compiling model
        print("compiling model...")
        opt = Adam(lr=0.000001)
        model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

    # load onlt the model
    if epochs == 0 and loadModelPath != None:
        model.summary()
        print("skipped training...")    
        return model

    # split data into sample and validation sets
    sample_images, validation_images = splitInstancesForTraining(train_images, dividor=datasetDividor)

    plotImagesClasses(sample_images, show=showPlot,name='classes_sample_instances')
    plotImagesClasses(validation_images, show=showPlot,name='classes_validation_instances')

    sample_features, sample_targets, sample_usage = getImagesAsCvDataLists(sample_images)
    validation_features, validation_targets, validation_usage = getImagesAsCvDataLists(validation_images)

    sample_features = np.array(sample_features) / 255
    validation_features = np.array(validation_features) / 255

    sample_features = tf.reshape(sample_features, (-1, image_shape[0], image_shape[1], 1))
    sample_targets = np.array(sample_targets)

    validation_features = tf.reshape(validation_features, (-1, image_shape[0], image_shape[1], 1))
    validation_targets = np.array(validation_targets)

    # training
    print("Training with n of", len(sample_features))
    print("validating with n of", len(validation_features))

    callbacks = []
    if useTensorBoard:
        log_path = 'tensorlog/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        print("started logging for tensorboard at in ", log_path)
        print("run command <tensorboard --logdir tensorlog/> and go to http://localhost:6006/ to follow training in browser")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))

    history = model.fit(sample_features,sample_targets,epochs = epochs , validation_data = (validation_features, validation_targets), callbacks=callbacks)

    plotHistory(history,epochs,show=showPlot,name='history_')

    # save the model
    if os.path.isfile(modelSavePath):
        modelSavePath = modelSavePath.replace('.keras', '')
        modelSavePath += datetime.now().strftime("%Y%m%d-%H%M%S") + '.keras'

    print("Saving model as", modelSavePath)
    model.save(modelSavePath)
    return model

def setPredictionsOnImages(_classifier, _images, usingCNN=False, image_shape=(48,48)):
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

def solution1(train_images, test_images):
    # train
    classifier = getRandomForestClassifier(train_images)

    # predict
    setPredictionsOnImages(classifier, test_images)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)
    

def solution2(train_images, test_images):
    # train
    classifier = getKNeighborsClassifier(train_images)

    # predict
    setPredictionsOnImages(classifier, test_images)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

def solution3(train_images, test_images):
    # train
    classifier = getCNNClassifier(train_images, datasetDividor=1.425,epochs=200,useTensorBoard=True)

    # predict
    setPredictionsOnImages(classifier, test_images, usingCNN=True)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

# main prog
def main():
    # read data
    train_images = readImagesFromCsv("resources/train.csv", max_n=100)
    test_images = readImagesFromCsv("resources/test.csv", max_n=10)

    # solution1(train_images, test_images)
    # solution2(train_images, test_images)
    solution3(train_images, test_images)

    sys.exit(0)

if __name__ == "__main__":
    main()