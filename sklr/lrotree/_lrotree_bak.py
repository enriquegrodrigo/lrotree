from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral

# Third party
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# Local application
from sklr.utils.validation import check_random_state
from sklr.clustering.clustering import ranking_clustering
from sklr.consensus import RankAggregationAlgorithm
from sklr.clustering._utils import is_complete, complete_rankings_nearest
from sklr.lrotree._utils import (
    DISTANCES, are_features_equal, are_rankings_equal
)

# =============================================================================
# LROTree estimator 
# =============================================================================


class LROTree(BaseEstimator, metaclass=ABCMeta):
    """
    LROTree
    """

    def __init__(self, 
                 min_samples_split = 2, 
                 max_depth = None, 
                 aggregation_algorithm = "borda_count", 
                 clustering_max_iters = 20, 
                 clustering_repeats = 10, 
                 clustering_mode_ranking_weight = 0.0001,
                 svc_regularization = 1.0, 
                 svc_kernel = 'rbf', 
                 svc_degree = 3, 
                 svc_gamma = 'scale', 
                 svc_coef0 = 0.0, 
                 svc_shrinking = True,
                 svc_cache_size = 200,
                 svc_max_iter = -1,
                 random_seed = 0 
        ):
        """Constructor."""
        # Initialize the hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.aggregation_algorithm = aggregation_algorithm
        self.clustering_max_iters = clustering_max_iters
        self.clustering_repeats = clustering_repeats
        self.clustering_mode_ranking_weight = clustering_mode_ranking_weight
        self.svc_regularization = svc_regularization
        self.svc_kernel = svc_kernel
        self.svc_degree = svc_degree
        self.svc_gamma = svc_gamma
        self.svc_coef0 = svc_coef0
        self.svc_shrinking = svc_shrinking
        self.svc_cache_size = svc_cache_size
        self.svc_max_iter = svc_max_iter
        self.random_seed = random_seed

        self.tree_model = None


    def get_depth(self):
        """Returns the depth of the decision tree.

        The depth of a tree is the maximum
        distance between the root and any leaf.
        """
        # Check if the tree is fitted
        #check_is_fitted(self)

        # Return the depth of the tree (stored in the
        # max_depth attribute of the underlying tree)
        return self.tree_model.max_depth

    def get_n_internal(self):
        """Returns the number of internal nodes of the decision tree."""
        # Check if the tree is fitted
        #check_is_fitted(self)

        # Return the number of internal nodes of the tree (stored
        # in the internal_count attribute of the underlying tree)
        return self.tree_model.internal_count

    def get_n_leaves(self):
        """Returns the number of leaves of the decision tree."""
        # Check if the tree is fitted
        #check_is_fitted(self)

        # Return the number of leaves of the tree (stored in
        # the leaf_count attribute of the underlying tree)
        return self.tree_model.leaf_count

    def get_n_nodes(self):
        """Returns the number of nodes of the decision tree."""
        # Check if the tree is fitted
        #check_is_fitted(self)

        # Return the number of nodes of the tree (just the
        # sum of the number of internal and leaf nodes)
        return self.tree_model.internal_count + self.tree_model.leaf_count

    def fit(self, X, Y):
        """Fit the decision tree on the training data and rankings."""
        # Validate the training data, the training
        # rankings and also the sample weights
        #(X, Y) = self._validate_data(X, Y, multi_output=True)

        (n_samples, n_classes) = Y.shape

        # Map to the input data type
        X = X.astype(np.float64)

        # Check the hyperparameters

        # Ensure that the maximum depth of the tree is greater than zero
        if self.max_depth is None:
            max_depth = np.iinfo(np.int32).max - 1
        else:
            if self.max_depth < 0:
                raise ValueError("max_depth must be greater than zero.")
            max_depth = self.max_depth

        # Ensure that the minimum number of samples to split an internal
        # node is a floating value greater than zero or less than or
        # equal one or an integer value greater than two
        if isinstance(self.min_samples_split, (Integral, np.integer)):
            if self.min_samples_split < 2:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]. "
                                 "Got the integer {}."
                                 .format(self.min_samples_split))
            min_samples_split = self.min_samples_split
        else:
            if self.min_samples_split <= 0 or self.min_samples_split > 1:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]. "
                                 "Got the float {}."
                                 .format(self.min_samples_split))
            min_samples_split = max(
                2,
                int(ceil(self.min_samples_split * n_samples)))

        # Initialize the builder
        builder = LROTreeBuilder(
                 min_samples_split = min_samples_split, 
                 max_depth = max_depth, 
                 aggregation_algorithm = self.aggregation_algorithm, 
                 clustering_max_iters = self.clustering_max_iters, 
                 clustering_repeats = self.clustering_repeats, 
                 clustering_mode_ranking_weight = self.clustering_mode_ranking_weight,
                 svc_regularization = self.svc_regularization, 
                 svc_kernel = self.svc_kernel, 
                 svc_degree = self.svc_degree, 
                 svc_gamma = self.svc_gamma, 
                 svc_coef0 = self.svc_coef0, 
                 svc_shrinking = self.svc_shrinking,
                 svc_cache_size = self.svc_cache_size,
                 svc_max_iter = self.svc_max_iter,
                 random_seed = self.random_seed 
        )

        # Build the tree
        self.tree_model = builder.build(X, Y)

        # Return it
        return self

    def predict(self, X):
        """Predict rankings for X.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y: np.ndarray of shape (n_samples, n_classes)
            The predicted rankings.
        """
        # Check the test data
        X = self._validate_data(X, reset=False)

        # Map to the input data type
        X = X.astype(np.float64)

        # Obtain the predictions using
        # the underlying tree structure
        predictions = self.tree_model.predict(X)

        # Return them
        return predictions




# =============================================================================
# Internal node
# =============================================================================
class InternalNode:
    """Representation of an internal node of a decision tree.

    Attributes
    ----------
    svc_model : 
       Support vector classification model to be used for the binary split. None for leaf nodes

    n_classes : int
        Number of different classes reaching the internal node.

    n_samples : int
        Number of training samples reaching the internal node.

    impurity : float
        Value of the splitting criterion at the internal node.

    consensus : np.ndarray of shape (n_classes,)
        Contains the constant prediction ranking of the internal node.
    """

    def __init__(
        self, svc_model, n_classes, n_samples, consensus
    ):
        """Constructor."""
        # SVC model for the split
        self.svc = svc_model
        
        # Initialize the number of samples 
        self.n_samples = n_samples
        self.n_classes = n_classes 


        # Initialize the impurity and
        # the consensus ranking
        self.consensus = consensus


# =============================================================================
# Internal node
# =============================================================================
class LeafNode:
    """Representation of a leaf node of a decision tree.

    Attributes
    ----------
    n_classes : int
        Number of different classes reaching the internal node.

    n_samples : int
        Number of training samples reaching the internal node.

    impurity : float
        Value of the splitting criterion at the internal node.

    consensus : np.ndarray of shape (n_classes,)
        Contains the constant prediction ranking of the internal node.
    """

    def __init__(
        self, n_classes, n_samples, consensus
    ):
        """Constructor."""
        # Initialize the number of samples 
        self.n_samples = n_samples
        self.n_classes = n_classes 


        # Initialize the impurity and
        # the consensus ranking
        self.consensus = consensus


# =============================================================================
# Tree Model 
# =============================================================================
class LROTreeModel:
    """Representation of a decision tree.

    Attributes
    ----------
    internal_count : int
        The number of internal nodes in the tree.

    leaf_count : int
        The number of leaf nodes in the tree.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    root : Node
        The root of the tree.

    children : list of Tree
        The children of the tree.
    """

    def __init__(self):
        """Constructor."""

        # Initialize the counter for internal node IDs, the
        # counter for leaf node IDs and the maximal tree depth
        self.internal_count = 0
        self.leaf_count = 0
        self.max_depth = 0

        # Initialize an empty list with
        # the children for this tree
        self.children = list()

    def _add_root(self, node):
        """Add a root to the tree."""
        # Add the root to the tree (directly copy
        # the reference and not the contents)
        self.root = node

        # Increase the number of
        # internal nodes or leaves
        if type(node) == InternalNode:
            self.internal_count += 1
        else:
            self.leaf_count += 1

    def _add_child(self, tree):
        """Add a child to the tree."""
        # Increase the total number
        # of internal nodes and leaves
        self.internal_count += tree.internal_count
        self.leaf_count += tree.leaf_count

        # Update the maximum depth of the tree (the maximum depth
        # of the tree is the maximum depth of their leaves)
        self.max_depth = max(self.max_depth, tree.max_depth + 1)

        # Add the child to the list
        self.children.append(tree)

    def predict(self, X):
        """Predict target for X."""
        return np.apply_along_axis(self._predict, 1, X)

    def _predict(self, x):
        """Predict target for x."""
        # If the root of the tree is a leaf
        # node, predict the consensus ranking
        if type(self.root) == LeafNode:
            return self.root.consensus
        # Otherwise, apply the decision path and recursively
        # predict the sample following the corresponding path
        else:
            # Use SVC model to select the children. 
            choosen_child = self.root.svc.predict([x])[0]
            
            # Recursively apply the children prediction
            pred_rank = self.children[choosen_child]._predict(x)
            
            # Check if predicted ranking needs completion
            if is_complete(pred_rank, self.root.consensus):
                return pred_rank
            else:
                # Child ranking is not complete, try to complete it with parent ranking
                pred_rank_set = np.array([pred_rank])
                complete_rankings_nearest(pred_rank_set, self.root.consensus)
                return pred_rank_set[0]



# =============================================================================
# Tree builder 
# =============================================================================

class LROTreeBuilder:
    """Build a decision tree."""

    def __init__(self, 
                 min_samples_split = 2, 
                 max_depth = None, 
                 aggregation_algorithm = "borda_count", 
                 clustering_max_iters = 20, 
                 clustering_repeats = 10, 
                 clustering_mode_ranking_weight = 0.001,
                 svc_regularization = 1.0, 
                 svc_kernel = 'rbf', 
                 svc_degree = 3, 
                 svc_gamma = 'scale', 
                 svc_coef0 = 0.0, 
                 svc_shrinking = True,
                 svc_cache_size = 200,
                 svc_max_iter = -1,
                 random_seed = 0 
        ):
        """Constructor."""
        # Initialize the hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.rank_algorithm = RankAggregationAlgorithm.get_algorithm(aggregation_algorithm)
        self.clustering_max_iters = clustering_max_iters
        self.clustering_repeats = clustering_repeats
        self.clustering_mode_ranking_weight = clustering_mode_ranking_weight
        self.svc_regularization = svc_regularization
        self.svc_kernel = svc_kernel
        self.svc_degree = svc_degree
        self.svc_gamma = svc_gamma
        self.svc_coef0 = svc_coef0
        self.svc_shrinking = svc_shrinking
        self.svc_cache_size = svc_cache_size
        self.svc_max_iter = svc_max_iter
        self.random_seed = random_seed

    def _build_leaf(self, tree, Y):
        """Build a leaf node."""
        # Initialize the number of classes and samples 
        n_classes = Y.shape[1]
        n_samples = Y.shape[0]

        # Initialize the impurity and the consensus ranking
        consensus = self.rank_algorithm.aggregate(Y)
        
        # Build the leaf node
        tree._add_root(LeafNode(n_classes, n_samples, consensus))

    def _build_internal(self, tree, X, Y, depth):
        """Build an internal node."""
        # Initialize the number of features and the number of
        # maximum splits from the stored values in the
        # corresponding structures of the splitter
        n_features = X.shape[1]
        
        # Initialize the number of classes (from
        # the information provided by the criterion)
        n_classes = Y.shape[1]

        # Initialize the number of samples
        n_samples = X.shape[0]

        # Initialize the impurity and the consensus ranking
        consensus = self.rank_algorithm.aggregate(Y)

        # Learn SVC model, with previously clustered data
        best_centroids, assigned_centroids, best_score, scores = ranking_clustering(
            2, 
            Y, 
            self.clustering_max_iters,
            self.clustering_repeats, 
            self.clustering_mode_ranking_weight,
            self.random_seed
        )
        
        y_clustered = assigned_centroids
        svc_model = svm.SVC(
            C = self.svc_regularization, 
            kernel = self.svc_kernel, 
            degree = self.svc_degree,
            gamma = self.svc_gamma, 
            coef0 = self.svc_coef0,
            shrinking = self.svc_shrinking,
            cache_size = self.svc_cache_size,
            max_iter = self.svc_max_iter
        )

        svc_model.fit(X, y_clustered)

        # Build the internal node
        tree._add_root(InternalNode(svc_model, n_classes, n_samples, consensus))

        # Build child nodes. It is a binary tree
        tree_child1 = LROTreeModel()
        self._build(tree_child1, X[assigned_centroids == 0], Y[assigned_centroids == 0], depth+1)
        tree._add_child(tree_child1)
        
        tree_child2 = LROTreeModel()
        self._build(tree_child2, X[assigned_centroids == 1], Y[assigned_centroids == 1], depth+1)
        tree._add_child(tree_child2)
        

    def _build(self, tree, X, Y, depth=0):
        """Recursively build a decision tree."""
        # Initialize the number of samples 
        n_samples = X.shape[0]

        # Check whether to create a leaf node. In this case,
        # a leaf node is created if the maximum depth has been
        # achieved, the number of samples is less than the
        # minimum number of samples to split an internal node,
        # all the values for each features are equal or all
        # the rankings are equal (properly managed missed classes)
        is_leaf = (
            depth == self.max_depth or
            n_samples < self.min_samples_split or
            are_features_equal(X) or
            are_rankings_equal(Y)
        )
       
        # Build the proper node (either a
        # leaf node or an internal node)
        if is_leaf:
            self._build_leaf(tree, Y)
        else:
            self._build_internal(tree, X, Y, depth)
        

    def build(self, X, Y):
        """Build a decision tree."""
        # Recursively build
        # the decision tree
        lrotree_model = LROTreeModel()
        self._build(lrotree_model, X, Y)
        return lrotree_model