import numpy as np
#from feature_extractor import FeatureExtractor as cifar10


# A decition tree node stores and individual node to be processed by the dection tree classifier, includes initialization and function to decide what direction the node will be placed on.
class DecitionTreeNode():
    def __init__(self, decision_function, feature_vector_index = None, threshold = None, left = None, right = None, info_gain = None, label = None): # Not sure if I'll keep 'info_gain'.
        self.feature_vector_index = feature_vector_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.label = label
        self.decision_function = decision_function

    def decide(self, feature):
        if self.label is not None:
            return self.label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else :   
            return self.right.decide(feature)
        
def gini_impurity_index(self, y):
    class_labels = np.unique(y) # return the sorted unique elements of an array (unique definition from np)
    gini = 0
    for cls in class_labels: # method that uses function for gini
        p_cls = len(y[y == cls]) / len(y)
        gini += p_cls**2
    return 1 - gini 

def gini_gain(previous_class, current_class):
    previous_gini_gain = gini_impurity_index(previous_class)  
    current_gini_gain = 0  
    previous_len = len(previous_class)  
    
    if len(current_class[0]) == 0 or len(current_class[1]) == 0:  
        return 0  
  
    for i in current_class:  
        current_length = len(i)  
        current_gini_gain += gini_impurity_index(i) * float(current_length) / previous_len  
  
    return previous_gini_gain - current_gini_gain




class DecitionTreeClassifier():
    def __init__(self, features, min_gain, min_samples, max_depth = 50):
        self.max_depth = max_depth
        self.features = features
        self.min_gain = min_gain
        self.min_samples = min_samples
        self.final_tree = None
        self.root = None

    def fit(self, x, y):
        self.n_class = np.unique(y).shape[0] # n_class: number of unique classes in y
        if self.max_features is None or self.max_features > X.shape[1]:
            self.max_features = x.shape[1]  # set max features to number of features in x
        self.feature_importance = np.zeros(X.shape[1])
        self.tree = build_tree(X, y, self.max_depth,
                                          self.min_gain, self.max_features,
                                          self.min_samples, self.n_class,
                                          self.feature_importance, x.shape[0])
        self.feature_importance /= np.sum(self.feature_importance)
        return self
    