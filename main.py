from sklearn.tree import DecisionTreeClassifier

from decision_tree_classifier import CustomDTC
from feature_extractor import FeatureExtractor
import numpy as np
from gaussian_naive_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from multi_layer_perceptron import MultiLayerPerceptron
import torch
import time


def main():
    extractor = FeatureExtractor()
    train_features_pca, train_labels, test_features_pca, test_labels = extractor.get_features_and_labels()

    train_features_pca = np.array(train_features_pca)
    test_features_pca = np.array(test_features_pca)
    train_labels = train_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    print("\nStarting the training process (Custom GNB)...")
    gnb = GaussianNaiveBayes()
    gnb.train_model(train_features_pca, train_labels)
    test_predictions = gnb.predict(test_features_pca)
    test_predictions = np.array(test_predictions)
    accuracy = np.mean(test_predictions == test_labels)
    print("Custom Naive Bayes Accuracy:", accuracy)

    print("\nStarting the training process (Scikit GNB)...")
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(train_features_pca, train_labels)
    sklearn_test_predictions = sklearn_gnb.predict(test_features_pca)
    sklearn_accuracy = accuracy_score(test_labels, sklearn_test_predictions)
    print("Scikit-learn Naive Bayes Accuracy:", sklearn_accuracy)

    # print("\nStarting the training process (Custom DTC)...")
    # dtc = CustomDTC()
    # dtc.train_model(train_features_pca, train_labels)
    # test_predictions_dtc = dtc.predict(test_features_pca)
    # test_predictions_dtc = np.array(test_predictions_dtc)
    # accuracy = np.mean(test_predictions_dtc == test_labels)
    # print("Custom Decision Tree Classifier:", accuracy)

    print("\nStarting the training process (Scikit DTC)...")
    sklearn_dtc = DecisionTreeClassifier()
    sklearn_dtc.fit(train_features_pca, train_labels)
    sklearn_dtc_test_predictions = sklearn_dtc.predict(test_features_pca)
    sklearn_dtc_accuracy = accuracy_score(test_labels, sklearn_dtc_test_predictions)
    print("Scikit-learn Decision Tree Accuracy: ", sklearn_dtc_accuracy)

    start_time = time.time()
    print("\nStarting the training process (MLP default layers config)...")
    mlp = MultiLayerPerceptron()
    mlp.train_model(torch.tensor(train_features_pca, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    mlp_accuracy = mlp.predict(torch.tensor(test_features_pca, dtype=torch.float32),
                               torch.tensor(test_labels, dtype=torch.long))
    end_time = time.time()
    print("MLP Accuracy:", mlp_accuracy)
    print(f"MLP Time (seconds): {end_time - start_time:.2f}")

    print("\nStarting the training process (MLP with -1 layer)...")
    mlp2 = MultiLayerPerceptron(depth=2)
    mlp2.train_model(torch.tensor(train_features_pca, dtype=torch.float32),
                     torch.tensor(train_labels, dtype=torch.long))
    mlp_accuracy = mlp2.predict(torch.tensor(test_features_pca, dtype=torch.float32),
                                torch.tensor(test_labels, dtype=torch.long))
    print("MLP2 Accuracy:", mlp_accuracy)

    print("\nStarting the training process (MLP with +1 layer)...")
    mlp3 = MultiLayerPerceptron(depth=4)
    mlp3.train_model(torch.tensor(train_features_pca, dtype=torch.float32),
                     torch.tensor(train_labels, dtype=torch.long))
    mlp_accuracy = mlp3.predict(torch.tensor(test_features_pca, dtype=torch.float32),
                                torch.tensor(test_labels, dtype=torch.long))
    print("MLP3 Accuracy:", mlp_accuracy)

    start_time = time.time()
    print("\nStarting the training process (MLP with -256 layers size - 256 total)...")
    mlp4 = MultiLayerPerceptron(hidden_layer_sizes=256)
    mlp4.train_model(torch.tensor(train_features_pca, dtype=torch.float32),
                     torch.tensor(train_labels, dtype=torch.long))
    mlp_accuracy = mlp4.predict(torch.tensor(test_features_pca, dtype=torch.float32),
                                torch.tensor(test_labels, dtype=torch.long))
    end_time = time.time()
    print("MLP4 Accuracy:", mlp_accuracy)
    print(f"MLP4 Time (seconds): {end_time - start_time:.2f}")

    start_time = time.time()
    print("\nStarting the training process (MLP with +512 layers size - 1024 total)...")
    mlp5 = MultiLayerPerceptron(hidden_layer_sizes=1024)
    mlp5.train_model(torch.tensor(train_features_pca, dtype=torch.float32),
                     torch.tensor(train_labels, dtype=torch.long))
    mlp_accuracy = mlp5.predict(torch.tensor(test_features_pca, dtype=torch.float32),
                                torch.tensor(test_labels, dtype=torch.long))
    end_time = time.time()
    print("MLP5 Accuracy:", mlp_accuracy)
    print(f"MLP5 Time (seconds): {end_time - start_time:.2f}")


if __name__ == "__main__":
    main()
