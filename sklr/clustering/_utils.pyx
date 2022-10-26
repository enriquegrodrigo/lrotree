# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# distutils: language = c++


# Third party
from libcpp.vector cimport vector

# Local application
from sklr._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D,
    INT64_t, INT64_t_1D, INT64_t_2D, SIZE_t)
from sklr._types cimport RANK_TYPE
from libc.stdio cimport printf



# =============================================================================
# Types
# =============================================================================

# =============================================================================
# Generic
# =============================================================================
ctypedef vector[INT64_t] BUCKET_t
ctypedef vector[BUCKET_t] BUCKETS_t

# Standard
from numbers import Integral, Real
from warnings import warn

# Third party
from libc.math cimport fabs
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

# Always include this statement after cimporting NumPy,
# to avoid possible segmentation faults
np.import_array()

# Local application
from sklr.utils._memory cimport copy_pointer_INT64_1D, copy_pointer_INT64_2D
from sklr.utils._ranking cimport rank_data_pointer
from sklr.utils._ranking cimport RANK_METHOD
from sklr.utils.ranking import (
    check_label_ranking_targets, check_partial_label_ranking_targets,
    _transform_rankings)
from sklr.utils.validation import (
    check_array, check_consistent_length, check_sample_weight)


# Epsilon value to add a little bit of helpful noise
cdef DTYPE_t EPSILON = np.finfo(np.float64).eps


cpdef bint is_complete(INT64_t_1D r, INT64_t_1D mode):
    cdef int n_classes = mode.shape[0]
    cdef int i
    cdef int j
    
    for i in range(n_classes):
        if (mode[i] == RANK_TYPE.TOP) or (mode[i] == RANK_TYPE.RANDOM):
            continue
        for j in range(n_classes):
            if mode[i] == r[j]: 
                # Found. We break to avoid wasting resources
                break
        else:
            # Not found, the rank is not complete taking into account the mode
            return False
        
    # All classes in the mode have been found
    return True

cpdef void complete_rankings_farthest(INT64_t_2D Y,
                                INT64_t_1D consensus):
        """Complete the rankings using the Borda count algorithm but by separating rankings."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef INT64_t n_ranked
        cdef INT64_t best_position
        cdef DTYPE_t best_distance
        cdef DTYPE_t local_distance
        cdef DTYPE_t *y

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k

        # Allocate memory for a ranking that will hold the
        # completed ranking for the corresponding iteration
        y = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Complete each ranking
        for sample in range(n_samples):
            # Reinitialize the number of
            # ranked classes in this ranking
            n_ranked = 0
            # As a preliminary step, compute the number
            # of ranked classes in this ranking
            for i in range(n_classes):
                if (Y[sample, i] != RANK_TYPE.RANDOM and
                        Y[sample, i] != RANK_TYPE.TOP):
                    n_ranked += 1
            # Check the classes that must be completed
            for i in range(n_classes):
                # Randomly missed classes can be at any
                # possible position, so that the one
                # minimizing the distance with respect
                # to the ranked classes is selected
                if Y[sample, i] == RANK_TYPE.RANDOM:
                    # Reinitialize the best position and
                    # the best distance for this class
                    best_position = 0
                    best_distance = 0
                    # Check the optimal position
                    # where the class must be inserted
                    # (j = 0 means before the first and
                    # j = m after the last label)
                    for j in range(n_classes + 1):
                        # Reinitialize the local distance
                        local_distance = 0.0
                        # Compute the distance w.r.t.
                        # the consensus ranking when
                        # this class is inserted between
                        # those on position j and j + 1
                        for k in range(n_classes):
                            # Only computes the distance
                            # for non missed classes
                            if Y[sample, k] != RANK_TYPE.RANDOM:
                                # Disagreement when inserting the class
                                # between those on position j and j + 1,
                                # increase the local distance
                                if (Y[sample, k] <= j and
                                        consensus[k] > consensus[i] or
                                        Y[sample, k] > j and
                                        consensus[k] < consensus[i]):
                                    local_distance += 1.0
                        # If the local distance is strictly more
                        # (because in case of a tie, the position
                        # with the smallest index is chosen)
                        # than the best distance found until now,
                        # change the best position and distance
                        if local_distance > best_distance:
                            best_position = j
                            best_distance = local_distance
                    # Insert the class in the best possible
                    # position according to the computed distance
                    y[i] = best_position
                # Top-k missed classes can only be at the latest positions
                # of the ranking. Therefore, set their positions to the
                # number of ranked items (plus one) and break the ties
                # according to the consensus ranking (since this will
                # minimize the distance with respect to it)
                elif Y[sample, i] == RANK_TYPE.TOP:
                    y[i] = n_ranked + 1
                # For ranked classes, directly copy
                # the position in the ranking
                else:
                    y[i] = Y[sample, i]

            # Add a little bit of noise based on the consensus ranking
            # to achieve that those classes inserted at the same position
            # are put in the same order than in the consensus ranking
            for i in range(n_classes):
                y[i] += EPSILON * consensus[i]

            # Rank again using the order of the consensus ranking to break
            # the ties when two classes inserted at the same position
            rank_data_pointer(
                data=y,
                y=&Y[sample, 0],
                method=RANK_METHOD.ORDINAL,
                n_classes=n_classes)

        # Once all the rankings have been completed,
        # release the allocated memory for the auxiliar
        # ranking since it is not needed anymore
        free(y)

cpdef void complete_rankings_nearest(INT64_t_2D Y,
                                INT64_t_1D consensus):
        """Complete the rankings using the Borda count algorithm but by separating rankings."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef INT64_t n_ranked
        cdef INT64_t best_position
        cdef DTYPE_t best_distance
        cdef DTYPE_t local_distance
        cdef DTYPE_t *y

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k

        # Allocate memory for a ranking that will hold the
        # completed ranking for the corresponding iteration
        y = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Complete each ranking
        for sample in range(n_samples):
            # Reinitialize the number of
            # ranked classes in this ranking
            n_ranked = 0
            # As a preliminary step, compute the number
            # of ranked classes in this ranking
            for i in range(n_classes):
                if (Y[sample, i] != RANK_TYPE.RANDOM and
                        Y[sample, i] != RANK_TYPE.TOP):
                    n_ranked += 1
            # Check the classes that must be completed
            for i in range(n_classes):
                # Randomly missed classes can be at any
                # possible position, so that the one
                # minimizing the distance with respect
                # to the ranked classes is selected
                if Y[sample, i] == RANK_TYPE.RANDOM:
                    # Reinitialize the best position and
                    # the best distance for this class
                    best_position = 0
                    best_distance = 0
                    # Check the optimal position
                    # where the class must be inserted
                    # (j = 0 means before the first and
                    # j = m after the last label)
                    for j in range(n_classes + 1):
                        # Reinitialize the local distance
                        local_distance = 0.0
                        # Compute the distance w.r.t.
                        # the consensus ranking when
                        # this class is inserted between
                        # those on position j and j + 1
                        for k in range(n_classes):
                            # Only computes the distance
                            # for non missed classes
                            if Y[sample, k] != RANK_TYPE.RANDOM:
                                # Disagreement when inserting the class
                                # between those on position j and j + 1,
                                # increase the local distance
                                if (Y[sample, k] <= j and
                                        consensus[k] > consensus[i] or
                                        Y[sample, k] > j and
                                        consensus[k] < consensus[i]):
                                    local_distance += 1.0
                        printf("%d %d %f\n", i, j, local_distance)
                        # If the local distance is less more
                        # (because in case of a tie, the position
                        # with the smallest index is chosen)
                        # than the best distance found until now,
                        # change the best position and distance
                        if local_distance < best_distance:
                            best_position = j
                            best_distance = local_distance
                    # Insert the class in the best possible
                    # position according to the computed distance
                    y[i] = best_position
                # Top-k missed classes can only be at the latest positions
                # of the ranking. Therefore, set their positions to the
                # number of ranked items (plus one) and break the ties
                # according to the consensus ranking (since this will
                # minimize the distance with respect to it)
                elif Y[sample, i] == RANK_TYPE.TOP:
                    y[i] = n_ranked + 1
                # For ranked classes, directly copy
                # the position in the ranking
                else:
                    y[i] = Y[sample, i]

            # Add a little bit of noise based on the consensus ranking
            # to achieve that those classes inserted at the same position
            # are put in the same order than in the consensus ranking
            for i in range(n_classes):
                y[i] += EPSILON * consensus[i]

            # Rank again using the order of the consensus ranking to break
            # the ties when two classes inserted at the same position
            rank_data_pointer(
                data=y,
                y=&Y[sample, 0],
                method=RANK_METHOD.ORDINAL,
                n_classes=n_classes)

        # Once all the rankings have been completed,
        # release the allocated memory for the auxiliar
        # ranking since it is not needed anymore
        free(y)