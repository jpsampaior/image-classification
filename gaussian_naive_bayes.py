import numpy as np


class GaussianNaiveBayes:
    def __init__(self):
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}

    def train_model(self, feature_vectors, labels):
        unique_classes = np.unique(labels)
        for class_label in unique_classes:
            # Filtra as amostras de que pertencem à classe atual
            class_samples = feature_vectors[labels == class_label]

            self.class_means[class_label] = class_samples.mean(axis=0)
            self.class_variances[class_label] = class_samples.var(axis=0) + 1e-6

            # número de amostras da classe / total de amostras
            self.class_priors[class_label] = class_samples.shape[0] / feature_vectors.shape[0]

    def gaussian_density(self, class_label, feature_vector):
        variance = self.class_variances[class_label]
        mean = self.class_means[class_label]
        exponent = np.exp(-((feature_vector - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent

    def predict(self, feature_vectors):
        predicted_classes = []

        for vector in feature_vectors:
            class_scores = {}

            for class_label in self.class_means:
                # Using log of probabilities, just like in Assignment 2
                log_prior = np.log(self.class_priors[class_label])
                log_likelihood = np.sum(np.log(self.gaussian_density(class_label, vector)))
                class_scores[class_label] = log_prior + log_likelihood

            predicted_class = max(class_scores, key=class_scores.get)
            predicted_classes.append(predicted_class)

        return np.array(predicted_classes)
