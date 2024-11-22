import numpy as np
#from feature_extractor import FeatureExtractor as cifar10


class DecitionTreeNode():
    def __init__(self, feature_vector_index = None, threshold = None, left = None, right = None, info_gain = None, value = None): # Not sure if I'll keep 'info_gain'.
        self.feature_vector_index = feature_vector_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecitionTreeClassifier():
    def __init__(self, max_depth = 50):
        self.max_depth = max_depth
        self.final_tree = None

    def predict():
    
    def fit():

    def build_tree():

    def calculate_gini():

    def info_gain():

    def find_best_split():

    def accuracy_score():
    

    

    



    

