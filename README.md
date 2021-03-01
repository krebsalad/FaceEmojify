# FaceEmojify
An application to predict the emotion on the face of a given person on a 48x48 picture

Used to learn machine learning...

# history
1. First test on 17-2-2012 to 18-2-2021 : trained and validated with data from a [kaggle competion](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules)

CNN configuration
* Convolutional 2D layer with 32 outputs and kernel size of 3
* Convolutional 2D layer with 64 outputs and kernel size of 3
* Dropout layer
* Regular densely connected NN layer with 128 outputs
* Final Regular densely connected layer with 7 outputs (0 - 6)

Results test1:
* 500EpochTestModel : training data was split 33% (sample) - 66% (validation)
* 1000EpochTestModel: built on top of 500EpochTestModel, training data was split 50% (sample) - 50% (validation)
* 750EpochTestModel: built on top of 500EpochTestModel, training data was split 33% (sample) - 66% (validation)
* 200EpochTestModel: built on top of 500EpochTestModel, training data was split 20% (sample) - 80% (validation)
* 50EpochTestModel: built on top of 500EpochTestModel, training data was split 20% (sample) - 80% (validation)
* 150EpochTestModel: built on top of 500EpochTestModel, training data was split 50% (sample) - 50% (validation)

2. Second test on 18-2-2021 - 19-2-2021 : trained and validated with data from a [kaggle competion](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules)
CNN configuration
* Convolutional 2D layer with 32 outputs and kernel size of 3
* Convolutional 2D layer with 64 outputs and kernel size of 3
* Dropout layer
* Convolutional 2D layer with 128 outputs and kernel size of 3
* Dropout layer
* Convolutional 2D layer with 256 outputs and kernel size of 3
* Dropout layer
* Regular densely connected NN layer with 128 outputs
* Final Regular densely connected layer with 7 outputs (0 - 6)

Results test2
* 500EpochTestModel : training data was split 66% (sample) - 33% (validation)
* 200EpochTestModel : training data was split 50% (sample) - 50% (validation)
* 350EpochTestModel : training data was split 70% (sample) - 30% (validation) with bias and kernel contraints with a max norm of 3
* 350EpochTestModel : training data was split 70% (sample) - 30% (validation) with bias and kernel contraints with a max norm of 3
* 500EpcochTestModel : training data was split 70% (sample) - 30% (validation) with bias and kernel contraints with a max norm of 3, instead using optimizer RMSprop and batch sizes of 32 

3. Third test on 25-2-2021 to 26-5-2021 : trained and validated with data from a [kaggle competion](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules)
CNN configuration
* Convolutional 2D layer with 32 outputs and kernel size of 3
* Drop out layer
* Convolutional 2D layer with 64 outputs and kernel size of 3
* Dropout layer
* Convolutional 2D layer with 128 outputs and kernel size of 3
* Dropout layer
* Convolutional 2D layer with 256 outputs and kernel size of 3
* Dropout layer
* Regular densely connected NN layer with 128 outputs
* Regular densely connected NN layer with 64 outputs
* Final Regular densely connected layer with 7 outputs (0 - 6)

Results test3
* 300EpochTestModel : training data was split 80% (sample) - 20% (validation)
* 200EpochTestModel : training data was split 80% - 20% and built on top of 300EpochTestModel
* 250EpochTestModel : training data was split 80% - 20% and built on top of 200EpochTestModel
* 500EpochTestModel : training data was split 80% - 20% and built on top of 250EpochTestModel
* 1000EpochTestModel : training data was split 80% - 20% and built on top of 500EpochTestModel
* 400EpochTestModel : training data was split 80% - 20% and built on top of 1000EpochTestModel

# Installation
1. Install [Git](https://git-scm.com/downloads), gitbash recommended aswell, and [python](https://www.python.org/)

2. Start a terminal and install pip
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```

3. Start a terminal, clone the project and go into the directory
```
cd ~/Desktop/projects/
git clone https://github.com/krebsalad/FaceEmojify.git
cd FaceEmojify
``` 

4. Create a [virtual environment](https://docs.python.org/3/library/venv.html) optionally...

5. Install sklearning, matplotlib, opencv, tensorflow, keras
```
pip install sklearn matplotlib jupyter mlxtend opencv-python tensorflow keras seaborn
```

6. Run the code with the following
```
python run.py
```

7. make changes directly to the script, go tot the bottom of the script 'run.py' to find a starting point

8. tensorboard can also be used
```
pip install tensorboard
```