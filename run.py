from FE_image_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import random
import threading

plt_lock = threading.Lock()

# for general use
def splitInstancesForTraining(train_instances, train_targets, splits=5):
    from sklearn.model_selection import StratifiedKFold

    # 100 as random seed for same results
    skf = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)

    folds = [([],[])] * splits
    i = 0
    for sample_indecies, validation_indecies in skf.split(train_instances, train_targets):
        for s in sample_indecies:
            folds[i][0].append(train_instances[s])
        for v in validation_indecies:
            folds[i][1].append(train_instances[v])
        i+=1

    return folds

def visualizeImageActivations(model, train_images, image_shape=(48,48), _show=False, _num_show_img=3):
    from keras.models import Model
    if not _show:
        return

    layer_outputs = [layer.output for layer in model.layers[:12]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    if len(train_images) > 2:
        indecies = random.sample(range(0, len(train_images)), _num_show_img)
        imgs = [train_images[i] for i in indecies]
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

def getLayerStack(num_chunks = 3, kern_size = 3, stride = 1, pad = 'valid'):
    from keras import Input
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, BatchNormalization, experimental, MaxPooling2D, GlobalMaxPooling2D

    # Normally a CNN architecture looks like the following: INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
    # For simplicity let's keep it now as INPUT -> [CONV with RELU -> POOL]*M -> FLATTEN-> [FC with RELU*K] -> FC

    # INPUT (images 48x48x1)
    input_layer = [Input(shape=(48, 48, 1))]

    # [CONV with RELU -> POOL] chunk loop
    chunks = num_chunks
    layer_stack = []
    for x in range(chunks):
        # CONV requires 4 hyperparameters (Number of Filters K (default=32, 64, 128, etc.), spatial extent/kernel size F (default=3), stride S (default=1), amount of zero padding P (default=valid))
        layer_stack += [Conv2D(filters=(32 * pow(2, x)), kernel_size=kern_size, strides=stride, activation='relu', padding=pad)]
        # POOL (In our case, default size 2))
        layer_stack += [MaxPooling2D(pool_size=2)]

    # Flatten layer
    flatten_layer = [Flatten()]

    # FC RELU layer loop
    fc_relu_stack = []
    for x in reversed(range(chunks)):
        fc_relu_stack += [Dense((32 * pow(2, x)), activation='relu')]

    # Finally, last classification layer (7 because we have 7 emotion classes)
    classification_layer = [Dense(7, activation="softmax")]

    cnn_stack = input_layer + layer_stack + flatten_layer + fc_relu_stack + classification_layer

    return cnn_stack

# evaluation util
def plotImagesClasses(_images, show=False, name='figure1', _classRange=7):
    classCount = [0] * _classRange
    for img in _images:
        if img.emotion != -1:             
            classCount[img.emotion] += 1
    print("class count ", classCount)
    with plt_lock:
        plt.bar(range(len(classCount)), classCount)
        save_path = 'images/'+ name + '.png'
        print("saving classes figures as", save_path)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.clf()
    return True

def plotHistory(history,epochs,show=False,name='figure2'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    with plt_lock:
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

        save_path = 'images/' + name + '.png'
        print("saving classes figures as", save_path)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.clf()
    return True

def plotRocCurve(model, test_features, test_targets, name='figure3', numOfClasses=7, _show=False):
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

    with plt_lock:
        for i in range(0, numOfClasses):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC class '+str(i)+' w area = '+str(roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        fig = plt.gcf()
        if _show:
            plt.show()
        save_path = 'images/'+ name + '.png'
        print("saving roc curve figure as", save_path)
        fig.savefig(save_path)
        plt.clf()

def plotConfusionMatrix(model, test_features, test_targets, name='figure4', usingCNN=False, _show=False):
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

    with plt_lock:
        sns.heatmap(conf_mat_norm, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        fig = plt.gcf()
        if _show:
            plt.show()
        save_path = 'images/'+ name + '.png'
        print("saving conf mat figures as", save_path)
        fig.savefig(save_path)
        plt.clf()

def plotPresionPlot(model, test_features, test_targets, name='figure5', numOfClasses=7, _show=False):
    from sklearn.metrics import precision_recall_curve, auc  
    from sklearn.preprocessing import label_binarize
    
    predictions = model.predict(test_features)
    
    test_targets_bin = label_binarize(test_targets, classes=[0,1,2,3,4,5,6])

    recall = dict()
    precision = dict()
    thresholds = dict()
    pres_auc = dict()
    for i in range(0, numOfClasses):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(test_targets_bin[:,i], predictions[:,i])
        pres_auc[i] = auc(recall[i], precision[i])

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in range(numOfClasses)]

    with plt_lock:
        for i in range(0, numOfClasses):
            plt.plot(recall[i], precision[i], color=colors[i], lw=1, marker='.', label='Logistic class' +str(i)+' w area = '+str(pres_auc[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")

        fig = plt.gcf()
        if _show:
            plt.show()
        save_path = 'images/'+ name + '.png'
        print("saving precision plot as", save_path)
        fig.savefig(save_path)
        plt.clf()

# classifier
def trainKNNClassifier(train_instances, modelName="model_knn", currentFoldIndex=0, n_folds=5,showPlot=False):
    from sklearn import neighbors
    # setup classifier
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)

    if not os.path.isdir('images/'+modelName):
        os.makedirs('images/'+modelName)

    # split data into sample and validation sets
    folds = splitInstancesForTraining(train_instances, getImagesEmotionsLists(train_instances), splits=n_folds) 
    sample_instances = folds[0][0]
    # validation_instances = folds[0][1]

    plotImagesClasses(sample_instances, show=showPlot,name=modelName+'/classes_train_fold_'+str(currentFoldIndex))

    sample_features, sample_targets, sample_usage = getImagesAsDataLists(sample_instances)
    # validation_features, validation_targets, validation_usage = getImagesAsDataLists(validation_instances)
    
    # train
    print("Training with n of", len(sample_features))
    classifier.fit(sample_features, sample_targets)
    return classifier

def trainCNNClassifier(train_images, layers=getLayersDefault(), currentFoldIndex=0, n_folds=5, epochs=500, image_shape=(48,48), modelSaveName='lastUsedModel', loadModelPath=None, showPlot=False, useTensorBoard=False):
    from keras.backend import clear_session as resetTraining
    from keras.models import Sequential
    from keras.optimizers import Adam, RMSprop

    resetTraining()

    if not os.path.isdir('images/'+modelSaveName):
        os.makedirs('images/'+modelSaveName)
    if not os.path.isdir('models/'+modelSaveName):
        os.makedirs('models/'+modelSaveName)

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
        print("No model given, creating new model from given layers definition...")
        print("adding layers...")
        
        model = Sequential()
        for l in layers:
            model.add(l)

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
    images_folds = splitInstancesForTraining(train_images, getImagesEmotionsLists(train_images), n_folds)

    sample_images, validation_images = images_folds[currentFoldIndex]

    plotImagesClasses(sample_images, show=showPlot,name=modelSaveName+'/classes_train_fold_'+str(currentFoldIndex))
    plotImagesClasses(validation_images, show=showPlot,name=modelSaveName+'/classes_val_fold_'+str(currentFoldIndex))

    sample_features, sample_targets, _ = getImagesAsCvDataLists(sample_images)
    validation_features, validation_targets, _ = getImagesAsCvDataLists(validation_images)

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
        log_path = 'tensorlog/' + modelSaveName + '/log_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
        print("started logging for tensorboard at in ", log_path)
        print("run command <tensorboard --logdir tensorlog/> and go to http://localhost:6006/ to follow training in browser")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))

    history = model.fit(sample_features,sample_targets,epochs = epochs , validation_data = (validation_features, validation_targets), batch_size=32, callbacks=callbacks)
    
    plotHistory(history,epochs,show=showPlot,name=modelSaveName+'/history_fold_'+str(currentFoldIndex))
    
    # save the model
    modelSavePath = 'models/' + modelSaveName + '/' + 'result_' + str(currentFoldIndex) + '.keras'
    if os.path.isfile(modelSavePath):
        modelSavePath = modelSavePath.replace('.keras', '')
        modelSavePath += datetime.now().strftime("%Y%m%d-%H%M%S") + '.keras'

    print("Saving model as", modelSavePath)
    model.save(modelSavePath)
    return model

def setPredictionsOnImages(_classifier, _images, model_name='model', usingCNN=False, image_shape=(48,48), max_n=0):
    for it, image in enumerate(_images):
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
        if(max_n != 0 and it > max_n-1):
            break

def evaluateModel(model, test_images, image_shape=(48,48), usingCNN=False, _show=False, name='model', fold_nr=0):
    test_features, test_targets, test_usage = [], [], []
    score = 0

    if usingCNN:
        test_features, test_targets, test_usage = getImagesAsCvDataLists(test_images)
        test_features = np.array(test_features) / 255
        test_features = tf.reshape(test_features, (-1, image_shape[0], image_shape[1], 1))
        test_targets = np.array(test_targets)
   
        print("Testing with n of", len(test_features))
        results = model.evaluate(test_features, test_targets)
        score = results[1]
        print(results)

        plotRocCurve(model, test_features, test_targets, numOfClasses=7, _show=_show, name=name+'/roc_cruve_fold_'+str(fold_nr))
        plotPresionPlot(model, test_features, test_targets, _show=_show, name=name+'/precision_plot_fold_'+str(fold_nr))

    else:
        test_features, test_targets, test_usage = getImagesAsDataLists(test_images)
        print("Testing with n of", len(test_features))
        score = model.score(test_features, test_targets)
        print("Classifier score:", str(score))

    plotConfusionMatrix(model, test_features, test_targets, _show=_show, usingCNN=usingCNN, name=name+'/confusion_mat_fold_'+str(fold_nr))
    return score

def main_KNN(train_images, test_images):
    # train
    classifier = trainKNNClassifier(train_images, 'modelKNN')

    # evaluate
    evaluateModel(classifier, test_images, name='modelKNN')

    # show images
    # setPredictionsOnImages(classifier, test_images, max_n=50)
    # showImages(test_images, _showPredictedEmotion=True)
    # writeImages(test_images, _showPredictedEmotion=True, max_n=50)

def main_CNN(train_images, test_images, threading=False, crossValidate=False, useThreading=False):
    # setup models
    model_params = []

    model1_params = [train_images, getLayersDefault(), 0, 5, 1, (48,48), 'model1', None,False,True]
    model_params.append(model1_params)

    # train function
    def trainCNNClassifierFuture(params):
        return trainCNNClassifier(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    # train and evaluate all models
    scores = []
    for p, params in enumerate(model_params):
        if not crossValidate:
            m = trainCNNClassifierFuture(params)
            score = evaluateModel(m, test_images, usingCNN=True, name=model_params[p][6])
            scores.append([score])
        else:
            s = []
            for i in range(0, params[3]):
                params[2] = i
                m = trainCNNClassifierFuture(params)
                score = evaluateModel(m, test_images, usingCNN=True, name=model_params[p][6], fold_nr=i)
                s.append(score)
            scores.append(s)
    
    for i, s in enumerate(scores):
        print("Score model " + str(i) + ":" + str(s) + ", average: " + str(sum(s)/len(s)))

    # show images
    # setPredictionsOnImages(classifier, test_images, usingCNN=True, max_n=50)
    # showImages(test_images, _showPredictedEmotion=True)
    # writeImages(test_images, _showPredictedEmotion=True, max_n=50)

# main prog
def main():
    # read data
    train_images = readImagesFromCsv("resources/train.csv", max_n=1000)
    images = readImagesFromCsv("resources/icml_face_data.csv")

    test_images = []
    for image in images:
        if image.usage == 'PrivateTest':
            test_images.append(image)
    
    main_CNN(train_images, test_images, crossValidate=True)
    sys.exit(0)

if __name__ == "__main__":
    main()