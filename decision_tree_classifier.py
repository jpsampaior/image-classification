import numpy as np
from node import Node

'''
The decition_tree_classifier file contains functions and classes for the purposes of carrying out logic of
decision tree using the gini index. Helper methods are listed at the top of the file which are used by the 
class CustomDTC which carries out the logic of the decision tree.
'''

# Calculation of gini impurity index based off of gini impurity formula.
def compute_gini_impurity(labels):
    class_counts = np.bincount(labels)
    probabilities = class_counts / len(labels)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Used by decision tree. Calculates gini gain for a potential split in the decision tree based off
# the labels of the parent, left, and right nodes.
def compute_gini_gain(parent_labels, left_labels, right_labels):
    parent_gini = compute_gini_impurity(parent_labels)
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    weighted_gini = (n_left / n) * compute_gini_impurity(left_labels) + \
                    (n_right / n) * compute_gini_impurity(right_labels)
    gini_gain = parent_gini - weighted_gini
    return gini_gain

# Function to split the entire dataset fed into 'build_tree', based off the training features
# training labels, and the feature index/threshold given in previous function.
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

# Helper function used to get most occuring feature for a given label.
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

'''
CustomDTC will first be initiallized with a default max depth of 50 (corresponding to the max depth of the tree) and
and a root node which is initialized to None, which will ultimatly hold the final calculated root used to access the
entire decision tree.
'''
class CustomDTC:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    # Takes in training feature vectors and labels from dataset to build tree.
    def train_model(self, feature_vectors, labels):
        self.root = self.build_tree(feature_vectors, labels, depth=0)

    # Predicts the class labels given a provided test features using the trained decision tree.
    def predict(self, feature_vectors):
        predicted_classes = []

        for feature in feature_vectors:
            tree = self.root
            predicted_classes.append(tree.decide(feature))
        return predicted_classes

    # Not sure if this is necissary. Will delete after testing.
    def predict_single(self, vector, node):
        if node.is_leaf:
            return node.label
        if vector[node.feature_index] < node.threshold:
            return self.predict_single(vector, node.left)
        else:
            return self.predict_single(vector, node.right)

    # Function contains logic for decision tree construction. Inputs are the training features and training labels provided.
    # The function recursivly constructs the decision tree based on gini gain splitting criteria and returns the root node
    # of the tree after partitioning the tree into right and left subtrees.
    def build_tree(self, features, labels, depth=0):

        # Initialize variables to store best feature, threshold, and gain.
        best_feature = None
        best_threshold = None
        best_gain = -1

        # If the length of a label is 0, return an None
        if len(labels) == 0:
            return None
        # If there is only 1 label provided, return a node with that label.
        elif len(labels) == 1:
            return Node(None, None, None, labels[0])

        # If all labels are the same, return a node with that label.
        elif np.all(labels[0] == labels[:]):
            return Node(None, None, None, labels[0])

        # If max depth is reached, return a node with the most occuring feature.
        elif depth == self.max_depth:
            return Node(None, None, None, get_most_occurring_feature(labels[0]))

        # Number of features in training features provided
        n_features = features.shape[1]
        for feature_index in range(n_features):

            # Identify unique thresholds for the a given feature.
            thresholds = np.unique(features[:, feature_index])
            thresholds_mean = np.mean(thresholds)
            labels_new = []

            # Splits data based off of threshold mean
            x_left, x_right, y_left, y_right = split_data(features, labels, feature_index, thresholds_mean)
            labels_new.append(y_left)
            labels_new.append(y_right)
            gain = compute_gini_gain(labels, y_left, y_right)

            # Updates the best gain, feature, and threshold if the current split improves gini gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = thresholds_mean
        
        # Final decision tree partition using the best feature found.
        X_left, X_right, y_left, y_right = split_data(features, labels, best_feature,
                                                      best_threshold)

        depth += 1

        # Recursively builds the right and left subtrees.
        right_tree = self.build_tree(np.array(X_right), np.array(y_right), depth)
        left_tree = self.build_tree(np.array(X_left), np.array(y_left), depth)

        return Node(left_tree, right_tree,
                    lambda feature: feature[best_feature] < best_threshold)
