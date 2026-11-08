import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier, gini, entropy

def test_gini_purity():
    """Test Gini impurity calculation for pure and perfectly mixed binary data."""
    
    # Gini impurity must be 0.0 for a completely pure set (all identical elements)
    assert gini(np.array([1, 1, 1])) == 0.0
    
    # Gini impurity must be 0.5 for a maximally mixed binary distribution (50/50 split)
    assert gini(np.array([0, 1])) == 0.5


def test_entropy_purity():
    """Test Shannon entropy calculation for pure and perfectly balanced binary data."""
    
    # Entropy must be 0.0 bits for a completely pure set
    assert entropy(np.array([0, 0, 0])) == 0.0
    
    # Entropy must be 1.0 bit for an evenly split binary distribution (50/50 split)
    assert np.isclose(entropy(np.array([0, 1])), 1.0)


def test_partial_fit_buffer_retraining():
    """Test the streaming buffer mechanism and periodic retraining behavior."""
    
    clf = DecisionTreeClassifier(max_depth=3, buffer_size=10)
    # Push the first chunk (6 samples), which is below the buffer_size threshold of 10
    X1, y1 = np.ones((6, 2)), np.ones(6, dtype=int)
    clf.partial_fit(X1, y1)
    # Verify that model training has not been triggered yet (root remains None)
    assert clf.root is None

    # Push the second chunk (4 samples), bringing the total accumulated samples to 10 (= buffer_size)
    X2, y2 = np.zeros((4, 2)), np.zeros(4, dtype=int)
    clf.partial_fit(X2, y2)
    
    # Verify that the tree has been successfully trained and the internal buffer has been cleared
    assert clf.root is not None
    assert len(clf.buffer_X) == 0
    assert len(clf.buffer_y) == 0