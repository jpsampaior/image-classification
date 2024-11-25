# COMP 472 - Artificial Intelligence
#### Students:
- João Pedro Sampaio Ribeiro - 40322458
- Alexis Bernier - 40245755

#### Project Description:
This project reunited the implementation of several AI models to train and predict images from the CIFAR-10 dataset. The models implemented were:
1. Custom Caussian Naive Bayes
2. Scikit-learn Gaussian Naive Bayes
3. Custom Decision Tree Classifier
4. Scikit-learn Decision Tree Classifier
5. Custom Multilayer Perceptron
6. Custom Convolutional Neural Network (VGG11)

## Description of each file
### 1. `main.py`
This file is the main file of the project. It is responsible to call a menu that allows the user to choose which model to train and predict the CIFAR-10 dataset. The user can also choose the pre-defined variant types that is available.

### 2. `feature_extractor.py`
This file is responsible for the preparation of the CIFAR-10 dataset. First, it loads the entire CIFAR-10 dataset and filters it to work with a predefined number of images: 500 training images per class and 100 testing images per class. After filtering, it creates data loaders for both the training and testing datasets.

Next, the pre-trained ResNet18 model is used to extract features from the images, generating a 512x1 feature vector for each RGB image. Before this feature extraction, the images are resized to 224x224x3 and normalized during the dataset loading process.

After using ResNet18, the feature vectors are further reduced to a size of 50 dimensions using PCA. These reduced features are then ready to be used for training and testing the models.

### 3. `gaussian_naive_bayes.py`
The gaussian_naive_bayes.py implements a probabilistic classifier based on the Naive Bayes algorithm, assuming that the features of each class follow a Gaussian (normal) distribution. The model is trained to calculate the mean, variance, and prior probability for each class, which are then used for classification.

The train_model method takes feature vectors and labels as input, calculating the mean, variance, and prior probability for each class and storing them in dictionaries.

The gaussian_density function computes the Gaussian probability density for a given feature vector and class, using the class-specific mean and variance.

The predict method classifies input feature vectors by calculating the probability for each class and selecting the one with the highest score. It returns the predicted class labels.

Finally, the get_accuracy method compares the predictions with the true labels to calculate the accuracy of the model. This class provides a simple and efficient implementation of Gaussian Naive Bayes, suitable for classification tasks where features follow a Gaussian distribution.


### 4. `multi_layer_perceptron.py`
The multi_layer_perceptron.py implements a configurable mlp. It allows customization of the input size, hidden layers, and depth, making it flexible for different datasets and problems.

The constructor initializes a sequential network with the specified number of layers. The first layer maps input features to a hidden layer size, applying a ReLU activation. Middle layers include batch normalization for stability, ReLU activation, and linear mappings between hidden units. The final layer maps hidden representations to the output size, which corresponds to the number of classes.

The forward method defines how data flows through the network, enabling efficient inference or training.

The train_model method trains the network using input features and labels. It employs cross-entropy loss as the objective function and SGD with momentum for optimization. Training occurs over a specified number of epochs, with gradients computed and parameters updated in each iteration.

The predict method evaluates the network on test data, using the trained model to compute class probabilities. It selects the class with the highest probability as the predicted label for each input.

Finally, the get_accuracy method calculates the proportion of correctly predicted labels compared to the true labels, providing a performance metric for the model. This class is a flexible and efficient implementation of a multi-layer perceptron for supervised learning tasks.

### 5. `vgg11.py`
The vgg11.py implements a configurable convolutional neural network inspired by the VGG-11 architecture. It allows customization of the kernel size, providing flexibility for different image processing tasks. The model comprises convolutional layers for feature extraction, followed by fully connected layers for classification.

The constructor defines the convolutional layers with ReLU activations, batch normalization for stability, and max-pooling for dimensionality reduction. These layers capture spatial hierarchies in the input data. Fully connected layers process the extracted features, using dropout to reduce overfitting and a final layer to produce class predictions.

The forward method defines the flow of data through the network. It automatically adjusts the input size for fully connected layers based on the output from the convolutional layers, ensuring compatibility with various input dimensions.

The train_vgg11_model method trains the network using cross-entropy loss and stochastic gradient descent (SGD) with momentum. It iterates over training data for a specified number of epochs, updating parameters to minimize loss.

The predict_vgg11_model method evaluates the model on test data, returning predictions and true labels for analysis. The get_accuracy_vgg11_model function computes the accuracy, measuring the proportion of correct predictions. This implementation offers a robust and adaptable VGG-11-based solution for supervised learning tasks in computer vision.

### 6. `decision_tree_classifier.py`
### 7. `node.py`

## How to run the project
To run this project, you need to have Python installed on your machine. So before running the project, make sure you have Python installed. If you don't have Python installed, you can download it from the official website: https://www.python.org/downloads/

After installing Python, you can clone this repository to your local machine.

Once you have the repository cloned, you can navigate to the project's root directory and run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

After installing the dependencies, please check if all the dependencies were installed correctly, by checking the files for any errors on the import statements.

Finally, you can run the project by executing the following command:

```bash
python main.py
```

This command will start the project and display a menu with the available models and variants. You can choose which model to train and predict the CIFAR-10 dataset by entering the corresponding number. The program will then execute the selected model and display the results, as well as some additional evaluation metrics.