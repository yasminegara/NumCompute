import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier, gini, entropy

def test_gini_purity():
    """Test Gini impurity calculation for pure and perfectly mixed binary data."""
    
    assert gini(np.array([1, 1, 1])) == 0.0
    
    assert gini(np.array([0, 1])) == 0.5


def test_entropy_purity():
    """Test Shannon entropy calculation for pure and perfectly balanced binary data."""
    
    assert entropy(np.array([0, 0, 0])) == 0.0
    
    assert np.isclose(entropy(np.array([0, 1])), 1.0)


def test_partial_fit_buffer_retraining():
    """Test the streaming buffer mechanism and periodic retraining behavior."""
    
    clf = DecisionTreeClassifier(max_depth=3, buffer_size=10)
    
    X1, y1 = np.ones((6, 2)), np.ones(6, dtype=int)
    clf.partial_fit(X1, y1)
    
    assert clf.root is None

    X2, y2 = np.zeros((4, 2)), np.zeros(4, dtype=int)
    clf.partial_fit(X2, y2)
    
    assert clf.root is not None
    assert len(clf.buffer_X) == 0
    assert len(clf.buffer_y) == 0
    
def test_decision_tree_prediction_and_entropy():
    """Test full prediction capability and entropy criterion choice for the tree."""
    
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=2, buffer_size=4)
    
    X_train = np.array([[1.0, 2.0], [1.5, 1.8], [8.0, 9.0], [8.5, 9.2]])
    y_train = np.array([0, 0, 1, 1])
    
    clf.partial_fit(X_train, y_train)
    assert clf.root is not None
    
    X_test = np.array([[1.2, 1.9], [8.2, 9.1]])
    preds = clf.predict(X_test)
    
    assert preds.shape == (2,)
    assert preds[0] == 0
    assert preds[1] == 1