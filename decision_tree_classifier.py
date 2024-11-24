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


def split_data(features, labels, feature_index, threshold):
    X_left = []
    X_right = []

    y_left = []
    y_right = []

    for i in range(len(features)):
        if float(features[i][feature_index]) <= threshold:
            X_left.append(features[i])
            y_left.append(labels[i])
        else:
            X_right.append(features[i])
            y_right.append(labels[i])

    return X_left, X_right, y_left, y_right


def get_most_occurring_feature(labels):
    frequency = {}
    for cls in labels:
        if cls in frequency:
            frequency[cls] += 1
        else:
            frequency[cls] = 1

    most_common = None
    max_count = -1
    for key, count in frequency.items():
        if count > max_count:
            most_common = key
            max_count = count

    return most_common


class CustomDTC:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    def train_model(self, feature_vectors, labels):
        self.root = self.build_tree(feature_vectors, labels, depth=0)

    def predict(self, feature_vectors):
        predicted_classes = []

        for feature in feature_vectors:
            tree = self.root
            predicted_classes.append(tree.decide(feature))
        return predicted_classes

    def predict_single(self, vector, node):
        if node.is_leaf:
            return node.label
        if vector[node.feature_index] < node.threshold:
            return self.predict_single(vector, node.left)
        else:
            return self.predict_single(vector, node.right)

    def build_tree(self, features, labels, depth=0):
        best_feature = None
        best_threshold = None
        best_gain = -1

        if len(labels) == 0:
            return None

        elif len(labels) == 1:
            return Node(None, None, None, labels[0])

        elif np.all(labels[0] == labels[:]):
            return Node(None, None, None, labels[0])

        elif depth == self.max_depth:
            return Node(None, None, None, get_most_occurring_feature(labels[0]))

        n_features = features.shape[1]
        for feature_index in range(n_features):

            thresholds = np.unique(features[:, feature_index])
            thresholds_mean = np.mean(thresholds)
            labels_new = []
            x_left, x_right, y_left, y_right = split_data(features, labels, feature_index, thresholds_mean)
            labels_new.append(y_left)
            labels_new.append(y_right)
            gain = compute_gini_gain(labels, y_left, y_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = thresholds_mean

        X_left, X_right, y_left, y_right = split_data(features, labels, best_feature,
                                                      best_threshold)

        depth += 1

        right_tree = self.build_tree(np.array(X_right), np.array(y_right), depth)
        left_tree = self.build_tree(np.array(X_left), np.array(y_left), depth)

        return Node(left_tree, right_tree,
                    lambda feature: feature[best_feature] < best_threshold)
