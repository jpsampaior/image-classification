'''
Node class which serves as the data stucture containing the left and right subtrees, decision function (gini), as well as labels.
'''
class Node:
    def __init__(self, left, right, decision_function, label=None):
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.label = label
    
    # Determines label given a feature vector.
    def decide(self, feature):
        if self.label is not None:
            return self.label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)
