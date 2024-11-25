from sklearn.tree import DecisionTreeClassifier
from decision_tree_classifier import CustomDTC
from feature_extractor import FeatureExtractor
from gaussian_naive_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from multi_layer_perceptron import MultiLayerPerceptron
import time
from vgg11 import *


def train_model_choice(choice, extractor, train_features_pca, train_labels, test_features_pca, test_labels):
    if choice == '1':
        print("\nStarting the training process (Custom GNB)...")
        gnb = GaussianNaiveBayes()
        gnb.train_model(train_features_pca, train_labels)
        gnb.predict(test_features_pca)
        accuracy = gnb.get_accuracy(test_labels)
        print("Custom Naive Bayes Accuracy:", accuracy)

    elif choice == '2':
        print("\nStarting the training process (Scikit GNB)...")
        sklearn_gnb = GaussianNB()
        sklearn_gnb.fit(train_features_pca, train_labels)
        sklearn_test_predictions = sklearn_gnb.predict(test_features_pca)
        sklearn_accuracy = accuracy_score(test_labels, sklearn_test_predictions)
        print("Scikit-learn Naive Bayes Accuracy:", sklearn_accuracy)

    elif choice == '3':
        print("\nStarting the training process (Custom DTC)...")
        dtc = CustomDTC()
        dtc.train_model(train_features_pca, train_labels)
        test_predictions_dtc = dtc.predict(test_features_pca)
        test_predictions_dtc = np.array(test_predictions_dtc)
        accuracy = np.mean(test_predictions_dtc == test_labels)
        print("Custom Decision Tree Classifier Accuracy:", accuracy)

    elif choice == '4':
        print("\nStarting the training process (Scikit DTC)...")
        sklearn_dtc = DecisionTreeClassifier()
        sklearn_dtc.fit(train_features_pca, train_labels)
        sklearn_dtc_test_predictions = sklearn_dtc.predict(test_features_pca)
        sklearn_dtc_accuracy = accuracy_score(test_labels, sklearn_dtc_test_predictions)
        print("Scikit-learn Decision Tree Accuracy: ", sklearn_dtc_accuracy)

    elif choice == '5':
        start_time = time.time()
        print("\nStarting the training process (MLP default layers config)...")
        mlp = MultiLayerPerceptron()
        mlp.train_model(train_features_pca, train_labels)
        mlp.predict(test_features_pca)
        mlp_accuracy = mlp.get_accuracy(test_labels)
        end_time = time.time()
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
        print("MLP5 Accuracy:", mlp_accuracy)
        print(f"MLP5 Time (seconds): {end_time - start_time:.2f}")

    elif choice == '10':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG11().to(device)
        print("\nStarting the training process (CNN/VGG11)...")
        train_vgg11_model(model, extractor.train_loader, device)
        predictions, true_labels = predict_vgg11_model(model, extractor.test_loader, device)
        accuracy = get_accuracy_vgg11_model(predictions, true_labels)
        print("CNN/VGG11 Accuracy:", accuracy)

    elif choice == '11':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model2 = VGG11().to(device)
        model2.conv_layers = nn.Sequential(*list(model2.conv_layers.children())[:-1])
        print("\nStarting the training process (CNN/VGG11) with -1 layer...")
        train_vgg11_model(model2, extractor.train_loader, device)
        predictions, true_labels = predict_vgg11_model(model2, extractor.test_loader, device)
        accuracy = get_accuracy_vgg11_model(predictions, true_labels)
        print("CNN/VGG11 variant 2 Accuracy:", accuracy)

    elif choice == '12':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model3 = VGG11().to(device)
        model3.conv_layers = nn.Sequential(
            *list(model3.conv_layers.children()),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Adiciona nova convolução
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        print("\nStarting the training process (CNN/VGG11) with +1 layer...")
        train_vgg11_model(model3, extractor.train_loader, device)
        predictions, true_labels = predict_vgg11_model(model3, extractor.test_loader, device)
        accuracy = get_accuracy_vgg11_model(predictions, true_labels)
        print("CNN/VGG11 variant 3 Accuracy:", accuracy)

    elif choice == '13':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG11(kernel_size=4).to(device)
        print("\nStarting the training process (CNN/VGG11) with 4 kernel size...")
        train_vgg11_model(model, extractor.train_loader, device)
        predictions, true_labels = predict_vgg11_model(model, extractor.test_loader, device)
        accuracy = get_accuracy_vgg11_model(predictions, true_labels)
        print("CNN/VGG11 variant 4 Accuracy:", accuracy)

    elif choice == '14':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG11(kernel_size=2).to(device)
        print("\nStarting the training process (CNN/VGG11) with 2 kernel size...")
        train_vgg11_model(model, extractor.train_loader, device)
        predictions, true_labels = predict_vgg11_model(model, extractor.test_loader, device)
        accuracy = get_accuracy_vgg11_model(predictions, true_labels)
        print("CNN/VGG11 variant 5 Accuracy:", accuracy)


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
        print("10. CNN / VGG11 with default configuration")
        print("11. CNN / VGG11 with with -1 layer")
        print("12. CNN / VGG11 with with +1 layer")
        print("13. CNN / VGG11 with with 2 kernel size")
        print("14. CNN / VGG11 with with 4 kernel size")
        print("0. Exit")

        choice = input("\nEnter the number of the model you want to train: ")
        if choice == '0':
            print("\nExiting...")
            break

        train_model_choice(choice, extractor, train_features_pca, train_labels, test_features_pca, test_labels)

        again = input("\nDo you want to train another model? (y/n): ")
        if again.lower() != 'y':
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()