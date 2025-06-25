import numpy as np
import pandas as pd
import graphviz
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class TreeNode:
    def __init__(self, is_categorical=False, feature_name=None, feature_index=None, 
                 split_value=None, left=None, right=None, label=None, gini=None):
        self.is_categorical = is_categorical
        self.feature_name = feature_name
        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.label = label
        self.gini = gini


    def is_leaf(self):
        return self.label is not None
    
 
class SemanticDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_impurity_decrease=0.0, max_leaf_nodes=None, max_features=None,
                 num_bins=10, special_values_json=None, random_state=None):
        self.max_depth = max_depth if max_depth is not None else 10
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.num_bins = num_bins
        self.special_values_json = special_values_json
        self.random_state=random_state
        self.tree = None
        self.classes_ = None
        self.n_features_in_ = None
        self.max_features_ = None
        self.n_leaf_nodes_ = 0

        if self.special_values_json is None:
            self.special_values_json = {}

        self._rng = np.random.default_rng(random_state)


    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.df = X.copy()
        else:
            self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self.n_features_in_ = len(self.df.columns)

        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_ = int(np.sqrt(self.n_features_in_))
            elif self.max_features == 'log2':
                self.max_features_ = int(np.log2(self.n_features_in_))
        else:
            self.max_features_ = self.max_features

        self.labels = y
        self.classes_ = sorted(set(y))
        data = np.column_stack((self.df.values, y))
        self.tree = self._build_tree(data, 0, 0)


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = [self._predict_row(row, self.tree) for row in X]
        
        return np.array(predictions)


    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        pred_probas = [self._predict_row_proba(row, self.tree) for row in X]

        return np.array(pred_probas)


    def _gini_impurity(self, groups, classes):
        n_instances = sum([len(group) for group in groups])
        gini = 0.0
        
        for group in groups:
            size = len(group)
            if size == 0:
                continue
        
            score = 0.0
            class_counts = Counter(row[-1] for row in group)
            
            for class_val in classes:
                p = class_counts[class_val] / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)

        return gini


    def _split_data(self, data, feature_index, value, is_categorical):
        left, right = [], []
        if is_categorical:
            for row in data:
                if str(row[feature_index]) == str(value):
                    right.append(row)
                else:
                    left.append(row)
        else:
            for row in data:
                if float(row[feature_index]) <= float(value):
                    left.append(row)
                else:
                    right.append(row)
        
        return left, right


    def _evaluate_split(self, data, feature_index, split_value, is_categorical, classes):
        left, right = self._split_data(data, feature_index, split_value, is_categorical)
        gini = self._gini_impurity([left, right], classes)

        return gini, {
            'feature_index': feature_index,
            'feature_name': self.df.columns[feature_index],
            'value': split_value,
            'groups': (left, right),
            'is_categorical': is_categorical
        }
    
    
    def _find_best_split(self, data):
        classes = list(set(row[-1] for row in data))
        best_gini = float('inf')
        best_split = None
        feature_indices = list(range(len(data[0]) - 1))

        if self.max_features is not None:
            feature_indices = self._rng.choice(feature_indices, size=self.max_features_, replace=False)

        N = len(data)
        gini_before = self._gini_impurity([data], classes)

        def evaluate_split(data, feature_index, split_value, is_categorical):
            groups = self._split_data(data, feature_index, split_value, is_categorical)
            gini = self._gini_impurity(groups, classes)
            
            return gini, {
                'feature_index': feature_index,
                'feature_name': self.df.columns[feature_index],
                'value': split_value,
                'groups': groups,
                'is_categorical': is_categorical
            }

        for feature_index in feature_indices:
            feature_name = self.df.columns[feature_index]
            special_vals = set(self.special_values_json.get(feature_name, {}).get("special_values", {}).keys())
            unique_values = set(row[feature_index] for row in data)
            all_values = [row[feature_index] for row in data]
            
            numeric_values = sorted([float(val) for val in all_values if str(val).split('.')[0] not in special_vals])
            
            if len(numeric_values) > 1:
                percentiles = np.unique(np.percentile(numeric_values, np.linspace(0, 100, self.num_bins + 1)[1:-1]))

                for p in percentiles:
                    midpoint = np.round(p, 3)
                    gini, split_info = evaluate_split(data, feature_index, midpoint, is_categorical=False)
                    left_data, right_data = split_info['groups']

                    N_t_L = len(left_data)
                    N_t_R = len(right_data)
                    impurity_after = (N_t_L / N) * self._gini_impurity([left_data], classes) \
                                   + (N_t_R / N) * self._gini_impurity([right_data], classes)
                    impurity_decrease = gini_before - impurity_after

                    if gini < best_gini and (impurity_decrease >= self.min_impurity_decrease):
                        best_gini = gini
                        best_split = split_info

            for value in unique_values:
                if str(value).split('.')[0] in special_vals:
                    gini, split_info = evaluate_split(data, feature_index, value, is_categorical=True)
                    left_data, right_data = split_info['groups']
                    N_t_L = len(left_data)
                    N_t_R = len(right_data)
                    impurity_after = (N_t_L / N) * self._gini_impurity([left_data], classes) \
                                   + (N_t_R / N) * self._gini_impurity([right_data], classes)
                    impurity_decrease = gini_before - impurity_after
                    
                    if gini < best_gini and (impurity_decrease >= self.min_impurity_decrease):
                        best_gini = gini
                        best_split = split_info

        return best_split


    def _build_tree(self, data, current_depth, current_leaf_nodes):
        classes = list(set(row[-1] for row in data))
        
        if (self.max_depth is not None and current_depth >= self.max_depth) or len(classes) == 1 or (self.max_leaf_nodes is not None and current_leaf_nodes >= self.max_leaf_nodes):
            distribution, gini = self._class_distribution_and_gini(data)
            self.n_leaf_nodes_ += 1
            return TreeNode(label=distribution, gini=round(gini, 4))

        if len(data) < self.min_samples_split:
            distribution, gini = self._class_distribution_and_gini(data)
            self.n_leaf_nodes_ += 1
            return TreeNode(label=distribution, gini=round(gini, 4))

        best_split = self._find_best_split(data)
        if not best_split or len(best_split['groups'][0]) < self.min_samples_leaf or len(best_split['groups'][1]) < self.min_samples_leaf:
            distribution, gini = self._class_distribution_and_gini(data)
            self.n_leaf_nodes_ += 1
            return TreeNode(label=distribution, gini=round(gini, 4))

        left_data, right_data = best_split['groups']

        left_distribution, _ = self._class_distribution_and_gini(left_data)
        right_distribution, _ = self._class_distribution_and_gini(right_data)

        left_pred_class = max(left_distribution, key=left_distribution.get)
        right_pred_class = max(right_distribution, key=right_distribution.get)

        if left_pred_class == right_pred_class:
            distribution, gini = self._class_distribution_and_gini(data)
            self.n_leaf_nodes_ += 1
            return TreeNode(label=distribution, gini=round(gini, 4))

        left_node = self._build_tree(left_data, current_depth + 1, current_leaf_nodes + 1)
        right_node = self._build_tree(right_data, current_depth + 1, current_leaf_nodes + 1)

        gini = self._gini_impurity([left_data, right_data], classes)

        return TreeNode(
            is_categorical=best_split['is_categorical'],
            feature_name=best_split['feature_name'],
            feature_index=best_split['feature_index'],
            split_value=best_split['value'],
            left=left_node,
            right=right_node,
            gini=round(gini, 4)
        )


    def _class_distribution_and_gini(self, data):
        counts = Counter(row[-1] for row in data)
        total = sum(counts.values())
        distribution = {label: count / total for label, count in counts.items()}
        
        gini = 1.0 - sum((count / total) ** 2 for count in counts.values())

        return distribution, gini


    def _predict_row(self, row, node):
        if node.is_leaf():
            return max(node.label, key=node.label.get)

        split_value = node.split_value
        feature_index = node.feature_index

        special_vals = set(self.special_values_json.get(self.df.columns[feature_index], {}).get("special_values", {}).keys())

        if node.is_categorical:
            if str(row[feature_index]) == str(split_value):
                return self._predict_row(row, node.right)
            else:
                return self._predict_row(row, node.left)
        else:
            if str(row[feature_index]) in special_vals:
                return self._predict_row(row, node.right)
            else:
                if float(row[feature_index]) <= float(split_value):
                    return self._predict_row(row, node.left)
                else:
                    return self._predict_row(row, node.right)


    def _predict_row_proba(self, row, node):
        if node.is_leaf():
            return np.array([node.label.get(cls, 0) for cls in self.classes_])

        split_value = node.split_value
        feature_index = node.feature_index

        special_vals = set(self.special_values_json.get(self.df.columns[feature_index], {}).get("special_values", {}).keys())

        if node.is_categorical:
            if str(row[feature_index]) == str(split_value):
                return self._predict_row_proba(row, node.right)
            else:
                return self._predict_row_proba(row, node.left)
        else:
            if str(row[feature_index]) in special_vals:
                return self._predict_row_proba(row, node.right)
            else:
                if float(row[feature_index]) <= float(split_value):
                    return self._predict_row_proba(row, node.left)
                else:
                    return self._predict_row_proba(row, node.right)
            

def print_tree(tree, tree_depth=None, print_gini=False):
    if tree_depth is None:
        tree_depth = tree.max_depth

    def _print_tree(node, depth=0, max_depth=tree_depth):
        if depth > max_depth:
            subtree_depth = compute_depth(node)
            print(f"{'|   ' * depth}|--- truncated branch of depth {subtree_depth}")
            return

        if print_gini:
            gini_node = f"(Gini: {node.gini:.4f})"
        else:
            gini_node = ''

        if node.is_leaf() and node.label:
            print(f"{'|   ' * depth}|--- Predict: {max(node.label, key=node.label.get)} {gini_node}")
        else:
            split_type = "Categorical" if node.is_categorical else "Numerical"
            if node.is_categorical:
                feature_info_left = f"{node.feature_name} != {node.split_value}"
                feature_info_right = f"{node.feature_name} == {node.split_value}"
            else:
                feature_info_left = f"{node.feature_name} <= {node.split_value}"
                feature_info_right = f"{node.feature_name} >  {node.split_value}"

            print(f"{'|   ' * depth}|--- [{split_type}] {feature_info_left} {gini_node}")
            if node.left:
                _print_tree(node.left, depth + 1, max_depth)
            if node.right:
                print(f"{'|   ' * depth}|--- [{split_type}] {feature_info_right} {gini_node}")
                _print_tree(node.right, depth + 1, max_depth)

    def compute_depth(node):
        if node.is_leaf():
            return 1
        left_depth = compute_depth(node.left) if node.left else 0
        right_depth = compute_depth(node.right) if node.right else 0
        return max(left_depth, right_depth) + 1

    _print_tree(tree.tree)


def plot_tree(tree, max_depth=None):
    dot = graphviz.Digraph()
    dot.attr('node', shape='box', fontsize='8')
    dot.attr('edge', arrowsize='0.5', penwidth='0.5')
    dot.attr('graph', nodesep='0.8')
    
    def add_nodes_edges(node, parent_name=None, depth=0):
        nonlocal dot

        if max_depth is not None and depth >= max_depth + 1:
            node_label = "(...)"
            node_name = f"Node{len(dot.body)}"
            dot.node(node_name, node_label, shape='box', style='filled', fillcolor='lightgray', fontsize='8')
            if parent_name is not None:
                dot.edge(parent_name, node_name)
            return

        if node.is_leaf() and node.label:
            node_label = f"Predict: {max(node.label, key=node.label.get)} (Gini: {node.gini})"
        else:
            split_operator = "==" if node.is_categorical else "<="
            split_value = node.split_value
            feature_name = node.feature_name
            node_label = f"{feature_name} {split_operator} {split_value}"

        node_name = f"Node{len(dot.body)}"
        dot.node(node_name, node_label, shape='box', fontsize='8')

        if parent_name is not None:
            dot.edge(parent_name, node_name)

        if not node.is_leaf():
            add_nodes_edges(node.left, node_name, depth + 1)
            add_nodes_edges(node.right, node_name, depth + 1)

    add_nodes_edges(tree.tree)

    return dot
