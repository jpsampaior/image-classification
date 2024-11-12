from feature_extractor import FeatureExtractor
import numpy as np
from gaussian_naive_bayes import GaussianNaiveBayes


def main():
    extractor = FeatureExtractor()
    train_features_pca, val_features_pca = extractor.process()

    train_labels = np.array([label for _, label in extractor.filtered_train_data])
    val_labels = np.array([label for _, label in extractor.filtered_val_data])

    gnb = GaussianNaiveBayes()
    gnb.train_model(train_features_pca, train_labels)

    val_predictions = gnb.predict(val_features_pca)

    accuracy = np.mean(val_predictions == val_labels)
    print("Naive Bayes Accuracy:", accuracy)


if __name__ == "__main__":
    main()
