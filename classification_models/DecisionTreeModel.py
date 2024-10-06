import numpy as np
import pandas as pd

class Node:
    """
    + Represets a node in the decision tree.
    + Each node has a left and right child, and a split value
    + Each node can be a leaf node or a decision node
    """

    def __init__(self, x=None, y=None, feature=None, threshold=None, feature_list=None):
        """
        Inicialization constructor
        
        Attributes:
        + x: set of features that the node will use to make a decision. (np.array)
        + y: set of labels that the node will use to make a decision. (np.array)
        + feature: feature choosen (int)
        + threshold: threshold value to split the data (float)
        + feature_list: list of features that can be used to split the data (list)
        + left: left child node (Node)
        + right: right child node (Node)
        """
        self.x = x
        self.y = y
        self.feature = feature
        self.threshold = threshold
        self.feature_list = feature_list
        self.left = None
        self.right = None

    def is_leaf(self):
        """
        Check if the node is a leaf node. If the node has no children, it is a leaf node
        Output: boolean
        """
        return self.left is None and self.right is None

    def is_terminal(self):
        """
        Check if all the labels in y are the same. If so, the node is a terminal node
        Output: boolean
        """
        return len(np.unique(self.y)) == 1

    def sample_size(self):
        """
        Return the number of samples in the node
        Output: int
        """
        return len(self.y)     

    def get_value(self):
        """
        Return the most common label in the node
        Output: int
        """
        return np.argmax(np.bincount(self.y)) #ojo

    def _calculate_entropy(self, indices, y):
        dic = {'0': 0, '1': 0}
        for i in indices:
            dic[str(y[i])] += 1

        entropy = 0
        for value in dic.values():
            if value != 0:
                prob = value / len(indices)
                entropy += -prob * np.log2(prob)
        return entropy

    def _y_entropy(self):
        """
        Calculate the entropy of the labels
        Input: y (np.array)
        Output: float
        """
        dic = {'0': 0, '1': 0}

        for i in self.y:
            dic[str(i)] += 1

        entropy = 0
        for key, value in dic.items():
            prob = value / len(self.y)
            entropy += -prob * np.log2(prob)

        return entropy

    def calculate_info_gain(self, feature, threshold):
        if feature not in self.x.columns:
            raise ValueError(f"Feature {feature} not found in x")

        left_indices = self.x[self.x[feature] <= threshold].index
        right_indices = self.x[self.x[feature] > threshold].index

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        entropy_left = self._calculate_entropy(left_indices, self.y)
        entropy_right = self._calculate_entropy(right_indices, self.y)

        entropy_total = (len(left_indices) / len(self.y) * entropy_left) + (
                len(right_indices) / len(self.y) * entropy_right)
        info_gain = self._y_entropy() - entropy_total

        return info_gain

    def best_split(self):
        best_feature = None
        best_threshold = None
        best_info_gain = float('-inf')

        for feature in self.feature_list:
            if feature in self.x.columns:
                threshold = np.mean(self.x[feature])
                info_gain = self.calculate_info_gain(feature, threshold)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


class DecisionTree:
    """
    + Decision tree classifier
    + The tree is built recursively by splitting the data at each node
    """

    def __init__(self, x_train, y_train, max_depth=3):
        """
        Inicialization constructor
        
        Atrributes:
        + x_train: set of features to train the model (np.array)
        + y_train: set of labels to train the model (np.array)
        + max_depth: maximum depth of the tree (int)
        + root: root node of the tree (Node)
        + 
        """
        self.x_train = x_train
        self.y_train = y_train
        self.max_depth = max_depth
        self.root = None
        self.minimum_samples = 5
        self.feature_list = x_train.columns.tolist()

    def fit(self):
        """
        Fit the model to the training data. The tree is built recursively by splitting the data at each node
        """
        self.root = self.insert(self.root)

    def insert(self, node):
        """
        Inserts a new node in the tree. If the node is not a leaf node, the data is split at the best feature and threshold
        Input: node (Node)
        Output: node (Node)
        """
        
        if node is None:
            node = Node(x=self.x_train, y=self.y_train, feature_list=self.x_train.columns.values.tolist())

        if not node.is_terminal() and node.sample_size() > self.minimum_samples and len(node.feature_list) > 0:
            best_feature, threshold = node.best_split()
            node.feature = best_feature
            node.threshold = threshold
            node.feature_list = self._get_new_feature_list(node)
            node = self._set_children(node)

            if node.left is not None and node.left.sample_size() > 0:
                node.left.feature_list = self._get_new_feature_list(node)
                self.insert(node.left)
            
            if node.right is not None and node.right.sample_size() > 0:
                node.right.feature_list = self._get_new_feature_list(node)
                self.insert(node.right)
        
        return node

    def _get_new_feature_list(self, node):
        """
        Return the list of features that can be used to split the data
        Input: node (Node)
        Output: list
        """
        #if node.feature is None:
        #    node.feature_list = self.x_train.columns.values.tolist()
        #    return node.feature_list

        current_feature = node.feature
        filtered_features = [feature for feature in node.feature_list if feature != current_feature]
        return filtered_features

    def _set_children(self, node):
        """
        Assign children nodes to the parameter node in base of the threshold. Filters the data and creates the left and right children nodes
        Input: node (Node)
        Output: node (Node)
        """

        x_filtered_left = node.x[node.x[node.feature] <= node.threshold].reset_index(drop=True)
        x_filtered_right = node.x[node.x[node.feature] > node.threshold].reset_index(drop=True)
        y_filtered_left = node.y[node.x[node.feature] <= node.threshold].reset_index(drop=True)
        y_filtered_right = node.y[node.x[node.feature] > node.threshold].reset_index(drop=True)

        node.left = Node(x=x_filtered_left, y=y_filtered_left, feature_list=node.feature_list)
        node.right = Node(x=x_filtered_right, y=y_filtered_right, feature_list=node.feature_list)

        return node

    def _predict_recursive(self, x_val, node):
        """
        Predict the label of a sample by traversing the tree recursively
        Input: x_val (np.array), node (Node)
        Output: int
        """
        if node.is_leaf():
            return node.get_value()
        
        current_feature = node.feature

        if x_val[current_feature].values <= node.threshold:
            return self._predict_recursive(x_val, node.left)
        elif x_val[current_feature].values > node.threshold:
            return self._predict_recursive(x_val, node.right)

    def predict(self, x_test):
        """
        Predict the labels of the test data
        Input: x_test (pd.DataFrame)
        Output: y_pred (np.array)
        """
        y_pred = []

        for i, row in x_test.iterrows():
            row_df = pd.DataFrame(row).T
            y_pred.append(self._predict_recursive(row_df, self.root))

        return np.array(y_pred)

    def print(self, node, depth=0):
        """
        Print the tree
        Input: node (Node), depth (int)
        """
        indent = "    " * depth
        if node is not None:
            if node.is_leaf():
                print(
                    f"{indent}Leaf: Samples={node.sample_size()}, Value={node.get_value()}")
            else:
                print(f"{indent}Node: Feature={node.feature} <= {node.threshold:.2f}, Info_Gain={node.calculate_info_gain(node.feature, node.threshold):.2f}, Samples={node.sample_size()}, Value={node.get_value()}")
                print(f"{indent}Left:")
                self.print(node.left, depth + 1)
                print(f"{indent}Right:")
                self.print(node.right, depth + 1)

    def hello(self):
        print("Hello from DecisionTreeModel")

