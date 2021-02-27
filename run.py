from FE_image_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import random

# for general use
def splitInstancesForTraining(train_instances, randomize=True, dividor=4):
    # get random indecies
    sample_indecies = []
    if randomize:
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

def visualizeImageActivations(model, train_images, image_shape=(48,48), _show=False, _num_show_img=3):
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers[:12]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    if len(train_images) > 2:
        indecies = random.sample(range(0, len(train_images)), _num_show_img)
        imgs = [train_images[i] for i in indecies]
        if not _show:
            return
        for img in imgs:
            temp = [img.getCvMat().copy()]
            temp = np.array(temp) / 255
            temp = tf.reshape(temp, (-1, image_shape[0], image_shape[1], 1))
            activations = activation_model.predict(temp)

            for act in activations:
                for im in act:
                    if len(im.shape) > 2:
                        if im.shape[2] == 1:
                            show_im = cv2.resize(im,image_shape,interpolation=cv2.INTER_CUBIC)
                            cv2.imshow('img', show_im)
                            cv2.waitKey(0)
                        else:
                            show_im = cv2.resize(im[:,:,0],image_shape,interpolation=cv2.INTER_CUBIC)
                            for i in range(1, im.shape[2]):
                                show_im = np.hstack((show_im, cv2.resize(im[:,:,i],image_shape,interpolation=cv2.INTER_CUBIC)))
                            cv2.imshow('img',show_im)
                            cv2.waitKey(0)
                    continue
            cv2.destroyAllWindows()

def getCNNClassifier(train_images, datasetDividor=5, epochs=500, image_shape=(48,48), modelSavePath='models/lastUsedModel.keras', loadModelPath=None, showPlot=False, useTensorBoard=False):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, BatchNormalization, experimental, MaxPooling2D
    from keras.constraints import max_norm
    from keras.optimizers import Adam, RMSprop

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
        
        model = Sequential([experimental.preprocessing.RandomFlip("horizontal", input_shape=(image_shape[0], image_shape[1], 1)), 
                                    experimental.preprocessing.RandomRotation(0.1), 
                                    experimental.preprocessing.RandomZoom(0.1)])
        model.add(Conv2D(32,(3,3), kernel_initializer='he_normal', padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(64,(3,3), kernel_initializer='he_normal', padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128,(3,3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(Dense(64,activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(7, activation="softmax"))

        # compiling model
        print("compiling model...")
        opt = RMSprop(lr=0.000001)
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

    # info
    model.summary()
    visualizeImageActivations(model,train_images, _show=showPlot)

    # training
    print("Training with n of", len(sample_features))
    print("validating with n of", len(validation_features))
    callbacks = []
    if useTensorBoard:
        log_path = 'tensorlog/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        print("started logging for tensorboard at in ", log_path)
        print("run command <tensorboard --logdir tensorlog/> and go to http://localhost:6006/ to follow training in browser")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))

    history = model.fit(sample_features,sample_targets,epochs = epochs , validation_data = (validation_features, validation_targets), batch_size=32, callbacks=callbacks)
    
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


def plotRocCurve(model, test_features, test_targets, name='roc_curve_', image_shape=(48, 48), numOfClasses=7, _show=False):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from scipy.optimize import curve_fit

    predictions = model.predict(test_features)
    test_targets_bin = label_binarize(test_targets, classes=[0,1,2,3,4,5,6])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, numOfClasses):
        fpr[i], tpr[i], _ = roc_curve(test_targets_bin[:,i], predictions[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(numOfClasses)]

    for i in range(0, numOfClasses):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC class '+str(i)+' w area = '+str(roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    fig = plt.gcf()
    if _show:
        plt.show()
    save_path = 'images/'+ name + ''+ datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
    print("saving roc curve figure as", save_path)
    fig.savefig(save_path)

def plotConfusionMatrix(model, test_features, test_targets, name='confusion_mat_', image_shape=(48,48), usingCNN=False, _show=False):
    import seaborn as sns
    
    # confisuion matrix
    predictions = model.predict(test_features)
    target_predictions = []
    if usingCNN:
        for res in predictions:
            highestProb = 0
            highestProbI = -1
            for i, prob in enumerate(res):
                if prob > highestProb:
                    highestProbI = i
                    highestProb = res[i]
            target_predictions.append(highestProbI)
    else:
        target_predictions = predictions

    
    conf_mat = tf.math.confusion_matrix(test_targets, target_predictions)
    conf_mat = tf.cast(conf_mat, dtype='float')
    conf_mat_norm = []
    for row in conf_mat:
        s = sum(row)
        prob_row = []
        for n in row:
            p = n/s
            prob_row.append(p)
        conf_mat_norm.append(prob_row)

    sns.heatmap(conf_mat_norm, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig = plt.gcf()
    if _show:
        plt.show()
    save_path = 'images/'+ name + ''+ datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
    print("saving conf mat figures as", save_path)
    fig.savefig(save_path)

def evaluateModel(model, test_images, image_shape=(48,48), usingCNN=False, _show=False):
    test_features, test_targets, test_usage = [], [], []

    if usingCNN:
        test_features, test_targets, test_usage = getImagesAsCvDataLists(test_images)
        test_features = np.array(test_features) / 255
        test_features = tf.reshape(test_features, (-1, image_shape[0], image_shape[1], 1))
        test_targets = np.array(test_targets)

        
        print("Testing with n of", len(test_features))
        results = model.evaluate(test_features, test_targets)
        print(results)

        plotRocCurve(model, test_features, test_targets, image_shape=image_shape, numOfClasses=7, _show=_show)

    else:
        test_features, test_targets, test_usage = getImagesAsDataLists(test_images)
        print("Testing with n of", len(test_features))
        score = model.score(test_features, test_targets)
        print("Classifier score:", str(score))

    plotConfusionMatrix(model, test_features, test_targets, image_shape=image_shape, _show=_show, usingCNN=usingCNN)
        

def main_KNN(train_images, test_images):
    # train
    image_shape= (48,48)
    classifier = getKNeighborsClassifier(train_images)

    # predict
    setPredictionsOnImages(classifier, test_images)

    # evaluate
    evaluateModel(classifier, test_images)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

def main_CNN(train_images, test_images):
    # train
    classifier = getCNNClassifier(train_images, loadModelPath='models/test3/500EpochTestModel.keras', datasetDividor=2,epochs=0,useTensorBoard=True,showPlot=False)

    # predict
    setPredictionsOnImages(classifier, test_images, usingCNN=True)

    # evaluate model
    evaluateModel(classifier, test_images, _show=True, usingCNN=True)

    # showImages(test_images, _showPredictedEmotion=True)
    writeImages(test_images, _showPredictedEmotion=True)

# main prog
def main():
    # read data
    train_images = readImagesFromCsv("resources/train.csv", max_n=100)
    images = readImagesFromCsv("resources/icml_face_data.csv")

    test_images = []
    max_n = 100
    n = 0
    for image in images:
        if image.usage == 'PrivateTest' or image.usage == 'PublicTest':
            test_images.append(image)
            n+=1
            if n > max_n:
                break
    
    main_CNN(train_images, test_images)
    sys.exit(0)

if __name__ == "__main__":
    main()