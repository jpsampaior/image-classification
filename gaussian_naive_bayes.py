import numpy as np


class GaussianNaiveBayes:
    def __init__(self):
        self.predictions = None
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}
        self.accuracy = None

    # Calculating the mean, variance and prior for each class
    def train_model(self, feature_vectors, labels):
        unique_classes = np.unique(labels)
        for class_label in unique_classes:
            class_samples = feature_vectors[labels == class_label]

            self.class_means[class_label] = class_samples.mean(axis=0)
            self.class_variances[class_label] = class_samples.var(axis=0)

            self.class_priors[class_label] = class_samples.shape[0] / feature_vectors.shape[0]

    # Applying the gaussian formula to get the density
    def gaussian_density(self, class_label, feature_vector):
        mean = self.class_means[class_label]
        variance = self.class_variances[class_label]
        std_dev = np.sqrt(variance)

        exponent = np.exp(-0.5 * ((feature_vector - mean) / std_dev) ** 2)
        density = (1 / (std_dev * np.sqrt(2 * np.pi))) * exponent

        return density

    # Using log prior and log likelihood to get the total probability for the class
    # Then select the max score as final prediction
    def predict(self, feature_vectors):
        predicted_classes = []

        for vector in feature_vectors:
            class_scores = {}

            for class_label in self.class_means:
                log_prior = np.log(self.class_priors[class_label])
                log_density = np.sum(np.log(self.gaussian_density(class_label, vector)))
                class_scores[class_label] = log_prior + log_density

            # Selects the class with the highest score (maximum likelihood) as the predicted class.
            predicted_class = max(class_scores, key=class_scores.get)
            predicted_classes.append(predicted_class)

        self.predictions = np.array(predicted_classes)

        return self.predictions

    # Comparing the predictions with the real labels to get the accuracy
    def get_accuracy(self, labels):
        self.accuracy = np.mean(self.predictions == labels)

        return self.accuracy
