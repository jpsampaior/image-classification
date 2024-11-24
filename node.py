class Node:
    def __init__(self, is_leaf=False, feature_index=None, threshold=None, left=None, right=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
