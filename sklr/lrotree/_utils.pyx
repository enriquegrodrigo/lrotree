# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# distutils: language = c++


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport exp, fabs, fmin, log
import numpy as np
cimport numpy as np
from libc.stdlib cimport free, malloc
# from libc.stdio cimport printf

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()

# Local application
from sklr.consensus cimport RankAggregationAlgorithm
from sklr._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D, DTYPE_t_4D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t, SIZE_t_1D, SIZE_t_2D, RANK_TYPE)


# =============================================================================
# Constants
# =============================================================================

# The number of iterations, the minimum theta, the maximum
# theta and a convergence value for the Mallows criterion
cdef INT64_t N_ITERS = 10
cdef DTYPE_t LOWER_THETA = -20.0
cdef DTYPE_t UPPER_THETA = 10.0
cdef DTYPE_t EPSILON = 1e-5

# Distance measures for the Mallows criterion
ctypedef enum DISTANCE:
    KENDALL
    TAU 

ctypedef enum NODE:
    INTERNAL,
    LEAF

# Distance measures that can be
# employed for the Mallows criterion
DISTANCES = {
    "kendall": 0,
    "tau": 1
}


# =============================================================================
# Imports
# =============================================================================


# =============================================================================
# Functions 
# =============================================================================

cpdef DTYPE_t _generalized_kendall_distance_fast(INT64_t_1D y_true, INT64_t_1D y_pred,
                                    BOOL_t normalize):
    """Fast version of Kendall distance."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y_true.shape[0]
    cdef INT64_t n_ranked_classes = 0

    # Define some variables to be employed
    cdef DTYPE_t dist

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class


    # Initialize the Kendall distance
    # between the rankings to zero
    dist = 0.0

    for f_class in range(n_classes - 1):
        for s_class in range(f_class + 1, n_classes):
            # Check only classes present on the ranking
            if (y_true[s_class] != RANK_TYPE.TOP and
                y_true[s_class] != RANK_TYPE.RANDOM and
                y_true[f_class] != RANK_TYPE.TOP and
                y_true[f_class] != RANK_TYPE.RANDOM 
                ):
                n_ranked_classes += 1
                # There exist a disagreement among the rankings
                # if the compared classes are in opposite order
                if (y_true[f_class] < y_true[s_class] and
                        y_pred[f_class] > y_pred[s_class] or
                        y_true[f_class] > y_true[s_class] and
                        y_pred[f_class] < y_pred[s_class]):
                    dist += 1

    if normalize:
        if n_ranked_classes > 1:
            dist /= n_ranked_classes * (n_ranked_classes-1) / 2

    return dist

cpdef DTYPE_t _generalized_tau_score_fast(INT64_t_1D y_true, INT64_t_1D y_pred,
                                    BOOL_t normalize):
    """Fast version of Kendall distance."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y_true.shape[0]
    cdef INT64_t n_ranked_classes = 0

    # Define some variables to be employed
    cdef DTYPE_t score

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class


    # Initialize the Kendall tau 
    # between the rankings to zero
    score = 0.0

    for f_class in range(n_classes - 1):
        for s_class in range(f_class + 1, n_classes):
            # Check only classes present on the ranking 
            if (y_true[s_class] != RANK_TYPE.TOP and
                y_true[s_class] != RANK_TYPE.RANDOM and
                y_true[f_class] != RANK_TYPE.TOP and
                y_true[f_class] != RANK_TYPE.RANDOM 
                ):
                n_ranked_classes += 1
                # There exist an agreement among the rankings
                # if the compared classes are in the same order
                if (y_true[f_class] < y_true[s_class] and
                        y_pred[f_class] < y_pred[s_class] or
                        y_true[f_class] > y_true[s_class] and
                        y_pred[f_class] > y_pred[s_class]):
                    score += 1.0
                else:
                    score -= 1.0

    if normalize:
        if n_ranked_classes > 1:
            score /= n_ranked_classes * (n_ranked_classes-1) / 2

    return score 


cpdef DTYPE_t impurity(INT64_t_2D Y, INT64_t_1D consensus, DISTANCE distance_method):
        """Compute the impurity using the distance criterion."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define some values to be employed
        cdef DTYPE_t error 
        cdef DTYPE_t impurity
        cdef DTYPE_t mean_errork 

        # Define the indexes
        cdef SIZE_t sample

        # Initialize the distance between the
        # rankings and the consensus ranking to zero
        error = 0.0

        # Compute the distance between the completed
        # rankings and its corresponding consensus
        for sample in range(n_samples):
            if distance_method == KENDALL:
                error += ((1.0 / n_samples)
                             * _generalized_kendall_distance_fast(
                                 y_true=Y[sample],
                                 y_pred=consensus,
                                 normalize=True))
            elif distance_method == TAU:
                error += -((1.0 / n_samples)
                             * _generalized_tau_score_fast(
                                 y_true=Y[sample],
                                 y_pred=consensus,
                                 normalize=True))
                
        # Mean distance/neg tau. The  
        impurity = error

        # Return the impurity
        return impurity


cpdef BOOL_t are_features_equal(DTYPE_t_2D X):
    """Check whether all the samples in X are equal
    (only considering the useful features)."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = X.shape[0]
    cdef INT64_t n_features = X.shape[1]

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t feature

    # Check whether the values for all features are equal
    # When a value is not equal, the process is stopped
    for feature in range(n_features):
        for sample in range(n_samples):
            if (X[0, feature] != X[sample, feature]):
                return False

    # We only reach here if every value of every feature is equal
    return True


cpdef BOOL_t are_rankings_equal(INT64_t_2D Y):
    """Check whether all the samples in Y are equal."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = Y.shape[0]
    cdef INT64_t n_classes = Y.shape[1]

    # Define some values to be employed
    cdef BOOL_t equal
    cdef INT64_t relation

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # The rankings are initialize as
    # if all the classes were equal
    equal = True

    # Check whether all the rankings are equal, properly
    # handling the cases where missed classes are found
    for f_class in range(n_classes - 1):
        for s_class in range(f_class + 1, n_classes):
            # Initialize the precedence relation between this
            # pair of classes being tested for consistency
            relation = RANK_TYPE.RANDOM
            # Check the precedence relation between
            # this pair of classes for all the samples
            for sample in range(n_samples):
                # Found one sample where both pair of classes occur
                if (Y[sample, f_class] != RANK_TYPE.RANDOM and
                        Y[sample, s_class] != RANK_TYPE.RANDOM):
                    # Check whether this sample is the first one
                    # where both pair of classes ocurr. If it is,
                    # store the precedence relation for checking
                    # it in the next samples
                    if relation == RANK_TYPE.RANDOM:
                        if (Y[sample, f_class] <
                                Y[sample, s_class]):
                            relation = -1
                        elif (Y[sample, f_class] ==
                                Y[sample, s_class]):
                            relation = 0
                        else:
                            relation = 1
                    # Otherwise, check the precedence relation
                    # of this pair of classes for this sample
                    else:
                        if relation == -1:
                            equal = (Y[sample, f_class] <
                                     Y[sample, s_class])
                        elif relation == 0:
                            equal = (Y[sample, f_class] ==
                                     Y[sample, s_class])
                        else:
                            equal = (Y[sample, f_class] >
                                     Y[sample, s_class])
                if not equal:
                    break
            if not equal:
                break
        if not equal:
            break

    # Return whether all
    # the rankings are equal
    return equal