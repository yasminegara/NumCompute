import numpy as np


def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini", buffer_size=500):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.impurity = gini if criterion == "gini" else entropy

        self.root = None
        self.buffer_X = []
        self.buffer_y = []
        self._default_label = 0

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        current_impurity = self.impurity(y)
        best_gain = 0.0
        best_feat, best_thresh = None, None

        for f in range(n_features):
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                mask = X[:, f] <= t
                n_left = np.sum(mask)
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                gain = current_impurity - (
                    (n_left / n_samples) * self.impurity(y[mask]) +
                    (n_right / n_samples) * self.impurity(y[~mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = t

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        unique_classes = np.unique(y)
        majority_class = np.bincount(y).argmax()

        if (depth >= self.max_depth or
                len(unique_classes) == 1 or
                n_samples < self.min_samples_split):
            return Node(label=majority_class)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return Node(label=majority_class)

        mask = X[:, feat] <= thresh
        left_child = self._build_tree(X[mask], y[mask], depth + 1)
        right_child = self._build_tree(X[~mask], y[~mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(y) > 0:
            self._default_label = np.bincount(y).argmax()
        self.root = self._build_tree(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk, dtype=int)

        self.buffer_X.append(X_chunk)
        self.buffer_y.append(y_chunk)

        total_buffered = sum(len(c) for c in self.buffer_y)
        if total_buffered >= self.buffer_size:
            X_all = np.vstack(self.buffer_X)
            y_all = np.concatenate(self.buffer_y)
            self.fit(X_all, y_all)
            self.buffer_X.clear()
            self.buffer_y.clear()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.root is None:
            return np.full(len(X), self._default_label)

        def _predict_row(x, node):
            while node.label is None:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.label

        return np.array([_predict_row(row, self.root) for row in X])