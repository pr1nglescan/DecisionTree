import numpy as np
import math

class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, label = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
    def __str__(self, level=0):
        ret = "\t"*level+repr(self)+"\n"
        for child in self.left, self.right:
            if child == None: continue
            ret += child.__str__(level+1)
        return ret
    def __repr__(self):
        if self.label == None: return "[Feature: " + str(self.feature) + ", Thresh: " + str(self.threshold) + "]"
        else: return "[Label: " + str(self.label) + "]"


class Tree():
    def __init__(self, root=None):
        self.root = root
    
    def traverse(self, point):
        node = self.root
        while (node.label == None):
            if point[node.feature] >= node.threshold:
                node = node.right
            else:
                node = node.left
        return node.label
    #def toString(self):
        #node = self.root

def entropy(cell):
    result = 0
    if len(cell) <= 1: return result

    labels = np.unique(cell[:, -1])
    for label in labels:
        pr = np.sum(cell[:, -1] == label) / len(cell)
        result -= pr * math.log(pr)
    return result

def split(cell, feature, thresh):
    left = []
    right = []
    for entry in cell:
        if entry[feature] >= thresh:
            right.append(entry)
        else:
            left.append(entry)
    return np.array(left), np.array(right)

def info_gain(cell, feature, thresh):
    left, right = split(cell, feature, thresh)
    prleft = len(left) / len(cell)
    prright = len(right) / len(cell)
    return entropy(cell) - prleft*entropy(left) - prright*entropy(right)

def optimal_split(cell):
    bestf = (0,0,0) #feature, value, gain
    for feature in range(0, len(cell[0]) - 1):
        values = np.unique(cell[:, feature])
        bestv = (0, 0) #value, gain
        for value in values: 
            gain = info_gain(cell, feature, value)
            if value == np.min(values) or gain > bestv[1]: bestv = (value, gain)
        if feature == 0 or bestv[1] > bestf[2]: bestf = (feature, bestv[0], bestv[1])
    return bestf[0], bestf[1], bestf[2]

def fit(cell): 
    feat, thresh, gain = optimal_split(cell)
    if gain > 0:
        left, right = split(cell, feat, thresh)
        nodel = fit(left)
        noder = fit(right)
        return Node(feature=feat, threshold=thresh, left=nodel, right=noder)
    labels = list(cell[:, -1])
    return Node(label=max(labels, key=labels.count))

def accuracy(classifier, test, labels):
    total = 0
    for i in range(0, len(test)):
        point = test[i]
        label = labels[i]
        if classifier.traverse(point) == label: total += 1
    return total / len(test)

class Random_Forest():
    def __init__(self, trees=[], theta=None, B=None):
        self.trees = trees
        self.theta = theta
        self.B = B
    def traverse(self, point): 
        votes = []
        for t in self.trees:
            votes.append(t.traverse(point))
        return max(votes, key=votes.count)
    
def fit_random_forest(data, theta, B):
    trees = []
    for i in range(0, B):
        sample = data[np.random.choice(data.shape[0], int(theta*data.shape[0]))]
        tree = Tree(fit(sample))
        trees.append(tree)
    return trees

'''cell = [
    [0.2, 0.4, 0],
    [1.3, 1.2, 1],
    [0.5, 1.5, 0],
    [1.6, 0.9, 1],
    [0.8, 0.2, 1],
    [1.1, 1.3, 0],
    [0.3, 1.0, 0],
    [1.4, 0.4, 1],
    [0.6, 0.8, 1],
    [0.9, 1.1, 0],
    [1.5, 1.6, 1],
    [0.4, 0.5, 0],
    [1.0, 0.6, 1],
    [0.7, 1.4, 0],
    [1.2, 0.3, 1]
]
cell = np.array(cell)

decisionTree = Tree(root=fit(cell))
print(traverse(decisionTree, [0,1]))
print(decisionTree.root)'''


        

