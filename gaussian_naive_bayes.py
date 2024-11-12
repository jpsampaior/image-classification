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

