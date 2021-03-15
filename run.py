from FE_image_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import random
import threading
import time

plt_lock = threading.Lock()

# for general use
def splitInstancesForTraining(train_instances, train_targets, splits=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

    original_splits = splits

    if splits == 1:
        splits = 2

    # 100 as random seed for same results
    skf = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)

    folds = [[[],[]]] * splits
    i = 0
    for sample_indecies, validation_indecies in skf.split(train_instances, train_targets):
        for s in sample_indecies:
            folds[i][0].append(train_instances[s])
        for v in validation_indecies:
            folds[i][1].append(train_instances[v])
        i+=1

    if original_splits == 1:
        return folds[0]
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

def getLayerStack(num_chunks = 2, num_conv2d_layers = 2, kern_size = 3, stride = 1, pad = 'valid', num_fc_layers = 1):
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
        for i in range(num_conv2d_layers):
            layer_stack += [Conv2D(filters=(32 * pow(2, x)), kernel_size=kern_size, strides=stride, activation='relu', padding=pad)]
        # POOL (In our case, default size 2))
        layer_stack += [MaxPooling2D(pool_size=2)]
        layer_stack += [Dropout(0.25)]

    # Flatten layer
    flatten_layer = [Flatten()]

    # FC RELU layer loop
    fc_relu_stack = []
    for x in range(num_fc_layers):
        fc_relu_stack += [Dense((32 * pow(2, (num_chunks - x))), activation='relu')]
        fc_relu_stack += [Dropout(0.5)]

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
    printLog("class count " + str(classCount))
    with plt_lock:
        plt.bar(range(len(classCount)), classCount)
        save_path = 'images/'+ name + '.png'
        printLog("saving classes figures as " + save_path)
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
        printLog("saving classes figures as " + save_path)
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
        printLog("saving roc curve figure as " + save_path)
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
        printLog("saving conf mat figures as " + save_path)
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
        printLog("saving precision plot as " + save_path)
        fig.savefig(save_path)
        plt.clf()

# classifier
def trainKNNClassifier(train_test_images, modelName="model_knn", fold_nr=0, n_folds=5,showPlot=False):
    from sklearn import neighbors
    # setup classifier
    from sklearn.multiclass import OneVsRestClassifier
    
    classifier = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=3))

    if not os.path.isdir('images/'+modelName):
        os.makedirs('images/'+modelName)

    # split data into sample and validation sets
    sample_instances = train_test_images[0]

    plotImagesClasses(sample_instances, show=showPlot,name=modelName+'/classes_train_fold_'+str(fold_nr))

    sample_features, sample_targets, sample_usage = getImagesAsDataLists(sample_instances)
    # validation_features, validation_targets, validation_usage = getImagesAsDataLists(validation_instances)
    
    # train
    printLog("Training with n of " + str(len(sample_features)))

    start_time = time.time()
    classifier.fit(sample_features, sample_targets)
    time_passed = time.time() - start_time

    printLog("Succesfully completed training of " + modelName + " in " + str(time_passed))
    return classifier

def trainCNNClassifier(train_test_images, layers=getLayerStack(), fold_nr=0, epochs=500, image_shape=(48,48), modelSaveName='lastUsedModel', loadModelPath=None, showPlot=False, useTensorBoard=False, clearMem=True):
    from keras.backend import clear_session
    from keras.models import Sequential
    from keras.optimizers import Adam, RMSprop
    from sklearn.decomposition import PCA

    clear_session()

    if not os.path.isdir('images/'+modelSaveName):
        os.makedirs('images/'+modelSaveName)
    if not os.path.isdir('models/'+modelSaveName):
        os.makedirs('models/'+modelSaveName)

    model = None
    if loadModelPath:
        if os.path.isfile(loadModelPath):
            printLog("loading model " + loadModelPath)
            model = tf.keras.models.load_model(loadModelPath)
        else:
            printLog("could not find model " + loadModelPath)
            printLog("exiting...")
            sys.exit(1)
    else:
        loadModelPath = None

    if loadModelPath == None:
        # creating model 
        printLog("No model given, creating new model from given layers definition...")
        printLog("adding layers...")
        
        model = Sequential()
        for l in layers:
            model.add(l)

        # compiling model
        printLog("compiling model...")
        opt = Adam(lr=0.000001)
        model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

    # load onlt the model
    if epochs == 0 and loadModelPath != None:
        printLog("Skipping training for model "+ modelSaveName + "_fold_" + str(fold_nr) + ", because epochs set to 0")
        printLog(getModelSummaryAsString(model))
        printLog("training skipped...")    
        return model

    # split data into sample and validation sets
    sample_images, validation_images = train_test_images

    plotImagesClasses(sample_images, show=showPlot,name=modelSaveName+'/classes_train_fold_'+str(fold_nr))
    plotImagesClasses(validation_images, show=showPlot,name=modelSaveName+'/classes_val_fold_'+str(fold_nr))

    sample_features, sample_targets, tmp = getImagesAsCvDataLists(sample_images)
    validation_features, validation_targets, tmp = getImagesAsCvDataLists(validation_images)

    # remove unused data
    if clearMem:
        release_list(sample_images)
        release_list(validation_images)
        release_list(tmp)
        release_list(train_test_images)

    # pca_dimensions = PCA()
    # pca_dimensions.fit(sample_features + validation_features)
    # cumsum = np.cumsum(pca_dimensions.explained_variance_ratio_)
    # dimensions = np.argmax(cumsum >= 0.95) + 1
    # printLog(str(dimensions))
    # pca = PCA(n_components=dimensions)
    # sample_features_reduced = pca.fit_transform(sample_features)
    # sample_features_compressed = pca.inverse_transform(sample_features_reduced)

    sample_features = np.array(sample_features) / 255
    validation_features = np.array(validation_features) / 255

    sample_features = tf.reshape(sample_features, (-1, image_shape[0], image_shape[1], 1))
    sample_targets = np.array(sample_targets)

    validation_features = tf.reshape(validation_features, (-1, image_shape[0], image_shape[1], 1))
    validation_targets = np.array(validation_targets)

    # info
    printLog("Setting up training for model "+ modelSaveName + "_fold_" + str(fold_nr))
    
    printLog(getModelSummaryAsString(model))
    visualizeImageActivations(model,sample_images, _show=showPlot)

    # training
    printLog("Training with n of " + str(len(sample_features)))
    printLog("validating with n of " + str(len(validation_features)))
    callbacks = []
    if useTensorBoard:
        log_path = 'tensorlog/' + modelSaveName + '/log_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
        printLog("started logging for tensorboard at in " + log_path)
        printLog("run command <tensorboard --logdir tensorlog/> and go to http://localhost:6006/ to follow training in browser")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0))
    
    start_time = time.time()
    history = model.fit(sample_features,sample_targets,epochs = epochs , validation_data = (validation_features, validation_targets), batch_size=32, callbacks=callbacks)
    time_passed = time.time() - start_time

    printLog("Succesfully completed training of " + modelSaveName + "_fold_" + str(fold_nr) + " in " + str(time_passed))

    plotHistory(history,epochs,show=showPlot,name=modelSaveName+'/history_fold_'+str(fold_nr))
    
    # save the model
    modelSavePath = 'models/' + modelSaveName + '/' + 'result_' + str(fold_nr) + '.keras'
    if os.path.isfile(modelSavePath):
        modelSavePath = modelSavePath.replace('.keras', '')
        modelSavePath += datetime.now().strftime("%Y%m%d-%H%M%S") + '.keras'

    printLog("Saving model as " + modelSavePath)
    model.save(modelSavePath)

    # save summary
    summarySavePath = 'models/' + modelSaveName + '/' + 'summary.txt'
    if fold_nr == 0:
        summarySaveFile = open(summarySavePath, "w+")
        summarySaveFile.close()

    writeToFile(summarySavePath, 'Summary ' + modelSaveName + ' fold ' + str(fold_nr) + ':\n' + getModelSummaryAsString(model) + '\n')

    return model

def setPredictionsOnImages(_classifier, _images, model_name='model', usingCNN=False, image_shape=(48,48), max_n=0):
    for it, image in enumerate(_images):
        # predict 
        if not usingCNN:
            p = _classifier.predict([image.pixels])[0]
            printLog(image.usage + " img " + str(image.id) + " predicted:" + str(p))
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
            printLog(image.usage + " img " + str(image.id) + "predicted:" + str(p) + "probs: " + str(res))    
        
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
   
        printLog("Testing with n of " + str(len(test_features)))
        results = model.evaluate(test_features, test_targets)
        score = results[1]
        printLog("Evaluation results:" + str(results))

        plotRocCurve(model, test_features, test_targets, numOfClasses=7, _show=_show, name=name+'/roc_cruve_fold_'+str(fold_nr))
        plotPresionPlot(model, test_features, test_targets, _show=_show, name=name+'/precision_plot_fold_'+str(fold_nr))

    else:
        test_features, test_targets, test_usage = getImagesAsDataLists(test_images)
        printLog("Testing with n of " + str(len(test_features)))
        score = model.score(test_features, test_targets)
        printLog("Classifier score:" + str(score))

    plotConfusionMatrix(model, test_features, test_targets, _show=_show, usingCNN=usingCNN, name=name+'/confusion_mat_fold_'+str(fold_nr))
    return score

def main_KNN(train_images, eval_images):
    # train
    images_fold = splitInstancesForTraining(train_images, getImagesEmotionsLists(train_images), 1)
    classifier = trainKNNClassifier(images_fold, 'modelKNN')

    # evaluate
    evaluateModel(classifier, eval_images, name='modelKNN')

    # show images
    # setPredictionsOnImages(classifier, eval_images, max_n=50)
    # showImages(eval_images, _showPredictedEmotion=True)
    # writeImages(eval_images, _showPredictedEmotion=True, max_n=50)

def main_CNN(train_images, eval_images, threading=False, crossValidate=False, useThreading=False):
    # setup models
    model_params = []

    # first parameter is left empty as train_images need to be split which depends on crossvalidation or not
    # parameter order: train_test_images, layers, fold_nr, epochs, image_shape, modelSaveName, loadModelPath, (optionals, can leave default): showPlot, useTensorBoard clearMem(True)
    model1_params = [[[],[]], getLayerStack(num_chunks = 2, num_conv2d_layers = 2, kern_size = 3, stride = 1, pad = 'valid', num_fc_layers = 1), 0, 10, (48,48), 'test_model', None]
    model_params.append(model1_params)

    # load a model and only evaluate it
    # model2_params = [[[],[]], [], 0, 0, (48,48), 'test_model_2', 'models/2000EpochTestModel.keras']   
    # model_params.append(model2_params)

    # train and evaluate all models
    scores = []
    for p, params in enumerate(model_params):
        # normal training if not cross validating or any training is done(when epochs is 0)
        if not crossValidate or not params[3]:
            params[0] = splitInstancesForTraining(train_images, getImagesEmotionsLists(train_images), 1)

            model = trainCNNClassifier(*params)

            score = evaluateModel(model, eval_images, usingCNN=True, name=model_params[p][5])
            scores.append([score])

            del model

        # cross validation
        else:
            s = []
            for i in range(0, 5):
                params[0] = splitInstancesForTraining(train_images, getImagesEmotionsLists(train_images), 5)[i]
                params[2] = i

                model = trainCNNClassifier(*params)

                score = evaluateModel(model, eval_images, usingCNN=True, name=model_params[p][5], fold_nr=i)
                s.append(score)

                del model

            scores.append(s)
    
    # show results
    for i, s in enumerate(scores):
        printLog("Score model " + str(i) + ":" + str(s) + ", average: " + str(sum(s)/len(s)))
    # show images
    # setPredictionsOnImages(classifier, eval_images, usingCNN=True, max_n=50)
    # showImages(eval_images, _showPredictedEmotion=True)
    # writeImages(eval_images, _showPredictedEmotion=True, max_n=50)

# main prog
def main():
    # read data
    train_images = readImagesFromCsv("resources/train.csv", max_n=5000, normalize_data_set=True)
    eval_images = readImagesFromCsv("resources/icml_face_data.csv", usage_skip_list=['PublicTest', 'Training'])
    
    main_CNN(train_images, eval_images, crossValidate=True)
    sys.exit(0)

if __name__ == "__main__":
    main()