import numpy as np
from node import Node


def compute_gini_impurity(labels):
    class_counts = np.bincount(labels)
    probabilities = class_counts / len(labels)
    gini = 1 - np.sum(probabilities ** 2)
    return gini


def compute_gini_gain(parent_labels, left_labels, right_labels):
    parent_gini = compute_gini_impurity(parent_labels)
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    weighted_gini = (n_left / n) * compute_gini_impurity(left_labels) + \
                    (n_right / n) * compute_gini_impurity(right_labels)
    gini_gain = parent_gini - weighted_gini
    return gini_gain


def majority_class(labels):
    return np.argmax(np.bincount(labels))


def split_data(features, labels, feature_index, threshold):
    left_indices = features[:, feature_index] < threshold
    right_indices = ~left_indices

    X_left, y_left = features[left_indices], labels[left_indices]
    X_right, y_right = features[right_indices], labels[right_indices]

    return X_left, X_right, y_left, y_right


def find_best_split(features, labels):
    best_feature = None
    best_threshold = None
    best_gain = -1

    n_features = features.shape[1]
    for feature_index in range(n_features):
        thresholds = np.unique(features[:, feature_index])
        for threshold in thresholds:
            x_left, x_right, y_left, y_right = split_data(features, labels, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain = compute_gini_gain(labels, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def stopping_criteria(depth, max_depth, labels, min_samples_split):
    if depth >= max_depth:
        return True
    if len(labels) < min_samples_split:
        return True
    if len(np.unique(labels)) == 1:
        return True
    return False


class CustomDTC:
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def train_model(self, feature_vectors, labels):
        self.root = self.build_tree(feature_vectors, labels, depth=0)

    def predict(self, feature_vectors):
        predicted_classes = []

        for vector in feature_vectors:
            predicted_class = self.predict_single(vector, self.root)
            predicted_classes.append(predicted_class)

        return np.array(predicted_classes)

    def predict_single(self, vector, node):
        if node.is_leaf:
            return node.label
        if vector[node.feature_index] < node.threshold:
            return self.predict_single(vector, node.left)
        else:
            return self.predict_single(vector, node.right)

    def build_tree(self, features, labels, depth):

        best_feature, best_threshold, _ = find_best_split(features, labels)
        X_left, X_right, y_left, y_right = split_data(features, labels, best_feature, best_threshold)

        left_tree = self.build_tree(X_left, y_left, depth + 1)
        right_tree = self.build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_tree,
            right=right_tree
        )
