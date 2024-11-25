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
from evaluation import generate_confusion_matrix_custom, generate_confusion_matrix_sklearn, extract_evaluation_metrics, generate_evaluation_table


def train_model_choice(choice, train_features_pca, train_labels, test_features_pca, test_labels):
    if choice == '1':
        print("\nStarting the training process (Custom GNB)...")
        gnb = GaussianNaiveBayes()
        gnb.train_model(train_features_pca, train_labels)
        gnb.predict(test_features_pca)
        accuracy = gnb.get_accuracy(test_labels)
        generate_confusion_matrix_custom(test_labels, gnb.predict(test_features_pca), "Custom GNB")
        print("Custom Naive Bayes Accuracy:", accuracy)

    elif choice == '2':
        print("\nStarting the training process (Scikit GNB)...")
        sklearn_gnb = GaussianNB()
        sklearn_gnb.fit(train_features_pca, train_labels)
        sklearn_test_predictions = sklearn_gnb.predict(test_features_pca)
        sklearn_accuracy = accuracy_score(test_labels, sklearn_test_predictions)
        generate_confusion_matrix_sklearn(test_labels, sklearn_gnb.predict(test_features_pca), "Scikit GNB")
        print("Scikit-learn Naive Bayes Accuracy:", sklearn_accuracy)

    elif choice == '3':
        print("\nStarting the training process (Custom DTC)...")
        dtc = CustomDTC()
        dtc.train_model(train_features_pca, train_labels)
        test_predictions_dtc = dtc.predict(test_features_pca)
        test_predictions_dtc = np.array(test_predictions_dtc)
        accuracy = np.mean(test_predictions_dtc == test_labels)
        generate_confusion_matrix_custom(test_labels, dtc.predict(test_features_pca), "Custom DTC")
        print("Custom Decision Tree Classifier Accuracy:", accuracy)

    elif choice == '4':
        print("\nStarting the training process (Scikit DTC)...")
        sklearn_dtc = DecisionTreeClassifier()
        sklearn_dtc.fit(train_features_pca, train_labels)
        sklearn_dtc_test_predictions = sklearn_dtc.predict(test_features_pca)
        sklearn_dtc_accuracy = accuracy_score(test_labels, sklearn_dtc_test_predictions)
        generate_confusion_matrix_custom(test_labels, sklearn_dtc.predict(test_features_pca), "Scikit DTC")
        print("Scikit-learn Decision Tree Accuracy: ", sklearn_dtc_accuracy)

    elif choice == '5':
        start_time = time.time()
        print("\nStarting the training process (MLP default layers config)...")
        mlp = MultiLayerPerceptron()
        mlp.train_model(train_features_pca, train_labels)
        mlp.predict(test_features_pca)
        mlp_accuracy = mlp.get_accuracy(test_labels)
        end_time = time.time()
        generate_confusion_matrix_custom(test_labels, mlp.predict(test_features_pca), "MLP default layers config")
        print("MLP Accuracy:", mlp_accuracy)
        print(f"MLP Time (seconds): {end_time - start_time:.2f}")

    elif choice == '6':
        start_time = time.time()
        print("\nStarting the training process (MLP with -1 layer)...")
        mlp2 = MultiLayerPerceptron(depth=2)
        mlp2.train_model(train_features_pca, train_labels)
        mlp2.predict(test_features_pca)
        mlp_accuracy = mlp2.get_accuracy(test_labels)
        end_time = time.time()
        generate_confusion_matrix_custom(test_labels, mlp2.predict(test_features_pca), "MLP with -1 layer")
        print("MLP2 Accuracy:", mlp_accuracy)
        print(f"MLP2 Time (seconds): {end_time - start_time:.2f}")

    elif choice == '7':
        start_time = time.time()
        print("\nStarting the training process (MLP with +1 layer)...")
        mlp3 = MultiLayerPerceptron(depth=4)
        mlp3.train_model(train_features_pca, train_labels)
        mlp3.predict(test_features_pca)
        mlp_accuracy = mlp3.get_accuracy(test_labels)
        end_time = time.time()
        generate_confusion_matrix_custom(test_labels, mlp3.predict(test_features_pca), "MLP with +1 layer")
        print("MLP3 Accuracy:", mlp_accuracy)
        print(f"MLP3 Time (seconds): {end_time - start_time:.2f}")

    elif choice == '8':
        start_time = time.time()
        print("\nStarting the training process (MLP with -256 layers size - 256 total)...")
        mlp4 = MultiLayerPerceptron(hidden_layer_sizes=256)
        mlp4.train_model(train_features_pca, train_labels)
        mlp4.predict(test_features_pca)
        mlp_accuracy = mlp4.get_accuracy(test_labels)
        end_time = time.time()
        generate_confusion_matrix_custom(test_labels, mlp4.predict(test_features_pca), "MLP with -256 layers size - 256 total")
        print("MLP4 Accuracy:", mlp_accuracy)
        print(f"MLP4 Time (seconds): {end_time - start_time:.2f}")

    elif choice == '9':
        start_time = time.time()
        print("\nStarting the training process (MLP with +512 layers size - 1024 total)...")
        mlp5 = MultiLayerPerceptron(hidden_layer_sizes=1024)
        mlp5.train_model(train_features_pca, train_labels)
        mlp5.predict(test_features_pca)
        mlp_accuracy = mlp5.get_accuracy(test_labels)
        end_time = time.time()
        generate_confusion_matrix_custom(test_labels, mlp5.predict(test_features_pca), "MLP with +512 layers size - 1024 total")
        print("MLP5 Accuracy:", mlp_accuracy)
        print(f"MLP5 Time (seconds): {end_time - start_time:.2f}")
    
    elif choice == '10':
        all_evaluation_metrics = []
        print("\nStarting the training process (all models)")
        print("\nTraining Custom Gaussian Naive Bayes (Custom GNB)")
        gnb = GaussianNaiveBayes()
        gnb.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, gnb.predict(test_features_pca), "Custom Gaussian Naive Bayes (Custom GNB)"))

        print("\nTraining Scikit-learn Gaussian Naive Bayes (SK-Learn GNB)")
        sklearn_gnb = GaussianNB()
        sklearn_gnb.fit(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, sklearn_gnb.predict(test_features_pca), "Scikit-learn Gaussian Naive Bayes (SK-Learn GNB)"))

        print("\nTraining Custom Decision Tree Classifier (Custom DTC)")
        dtc = CustomDTC()
        dtc.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append( extract_evaluation_metrics(test_labels, dtc.predict(test_features_pca), "Custom Decision Tree Classifier (Custom DTC)"))

        print("\nTraining Scikit-learn Decision Tree Classifier (SK-Learn DTC)")
        sklearn_dtc = DecisionTreeClassifier()
        sklearn_dtc.fit(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, sklearn_dtc.predict(test_features_pca), "Scikit-learn Decision Tree Classifier (SK-Learn DTC)"))

        print("\nTraining Multi Layer Perceptron (MLP) with default configuration")
        mlp = MultiLayerPerceptron()
        mlp.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, mlp.predict(test_features_pca), "Multi Layer Perceptron (MLP) with default configuration"))
        
        print("\nTraining Multi Layer Perceptron (MLP) with -1 layer")
        mlp2 = MultiLayerPerceptron(depth=2)
        mlp2.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, mlp2.predict(test_features_pca), "Multi Layer Perceptron (MLP) with -1 layer"))

        print("\nTraining Multi Layer Perceptron (MLP) with +1 layer")
        mlp3 = MultiLayerPerceptron(depth=4)
        mlp3.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, mlp3.predict(test_features_pca), "Multi Layer Perceptron (MLP) with +1 layer"))

        print("\nTraining Multi Layer Perceptron (MLP) with -256 layer size (total 256)")
        mlp4 = MultiLayerPerceptron(hidden_layer_sizes=256)
        mlp4.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, mlp4.predict(test_features_pca), "Multi Layer Perceptron (MLP) with -256 layer size (total 256)"))

        print("\nTraining Multi Layer Perceptron (MLP) with +512 layer size (total 1024)")
        mlp5 = MultiLayerPerceptron(hidden_layer_sizes=1024)
        mlp5.train_model(train_features_pca, train_labels)
        all_evaluation_metrics.append(extract_evaluation_metrics(test_labels, mlp5.predict(test_features_pca), "Multi Layer Perceptron (MLP) with +512 layer size (total 1024)"))

        generate_evaluation_table(all_evaluation_metrics)


def main():
    extractor = FeatureExtractor()
    train_features_pca, train_labels, test_features_pca, test_labels = extractor.get_features_and_labels()

    train_features_pca = np.array(train_features_pca)
    test_features_pca = np.array(test_features_pca)
    train_labels = train_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    while True:
        print("\nChoose a model to train and predict:")
        print("1. Custom Gaussian Naive Bayes (Custom GNB)")
        print("2. Scikit-learn Gaussian Naive Bayes (SK-Learn GNB)")
        print("3. Custom Decision Tree Classifier (Custom DTC)")
        print("4. Scikit-learn Decision Tree Classifier (SK-Learn DTC)")
        print("5. Multi Layer Perceptron (MLP) with default configuration")
        print("6. Multi Layer Perceptron (MLP) with -1 layer")
        print("7. Multi Layer Perceptron (MLP) with +1 layer")
        print("8. Multi Layer Perceptron (MLP) with -256 layer size (total 256)")
        print("9. Multi Layer Perceptron (MLP) with +512 layer size (total 1024)")
        print("10. Train all models and display summary evaluation table (accuracy, precision, recall, F1-Measure)")
        print("0. Exit")

        choice = input("\nEnter the number of the model you want to train: ")
        if choice == '0':
            print("\nExiting...")
            break

        train_model_choice(choice, train_features_pca, train_labels, test_features_pca, test_labels)

        again = input("\nDo you want to train another model? (y/n): ")
        if again.lower() != 'y':
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
