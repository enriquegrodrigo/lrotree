from sklr.tree._criterion import _generalized_kendall_distance_fast
import numpy as np
import functools
from sklr.consensus import RankAggregationAlgorithm

from sklr.clustering._utils import is_complete, complete_rankings_farthest
import sklr.lrotree.skrandom as skrandom

def ranking_clustering(
    K, ranking_set, max_iters, repeats, mode_ranking_weight, balancing=False 
):
    """
    Clusters ranking_set in K clusters using a Kmeans++ initialization and several repeats
    """
    #centroids, ranking_centroid_distances, mode_rank = initialization(ranking_set, K, seed=0)
    best_score = -1.1
    best_centroids = []
    assigned_centroids = []
    scores = []
    for r in range(repeats):
        scores_r = []
        centroids, ranking_centroid_distances, mode_rank = initialization(ranking_set, K)
        for _ in range(max_iters):
            if balancing:
                rankings_assigned_centroid = assign_cluster_balancing(ranking_centroid_distances, ranking_set)
            else:
                rankings_assigned_centroid = assign_cluster(ranking_centroid_distances)

            new_centroids = update_centroids(ranking_set, rankings_assigned_centroid, mode_rank, K, mode_ranking_weight)
            ranking_centroid_distances = np.array([distances_from_centroid(ranking_set, centroid) for centroid in centroids]).T

            if np.array_equal(centroids, new_centroids):
                # Update does not change the centroids, we have reached convergence, so we stop early
                break
            else:
                centroids = new_centroids

        score =  incomplete_rank_cluster_eval(ranking_set, rankings_assigned_centroid, centroids)
        scores_r.append(score)
        if score > best_score:
            best_centroids = centroids 
            best_score = score 
            assigned_centroids = rankings_assigned_centroid 

    return best_centroids, assigned_centroids, best_score, scores

# HELPER functions

def initialization(incomplete_ranking_set, K):
    """
    Initializes centroids by farthest distance from the previous rank
    """
    # We obtain the aggregate ranking for the set
    rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")
    mode_rank = rank_algorithm.aggregate(incomplete_ranking_set)
    
    # We obtain the first complete ranking at random
    centroids = []
    centroids.append(random_first_rank_initialization(incomplete_ranking_set, mode_rank))
    
    # We obtain the rest of the centroids sampling by farthest distance from the previous rank
    distances_from_centroids = []
    for k in range(K-1):
        distances = distances_from_centroid(incomplete_ranking_set, centroids[-1])
        distances_from_centroids.append(distances)
        next_centroid_incomplete = farther_ranking_sample(incomplete_ranking_set, distances)
        next_centroid = complete_ranking_with_reference(next_centroid_incomplete, centroids[-1])
        centroids.append(next_centroid)
    
    distances = distances_from_centroid(incomplete_ranking_set, centroids[-1])
    distances_from_centroids.append(distances)
        
    return np.array(centroids), np.array(distances_from_centroids).T, mode_rank


def update_centroids(incomplete_ranking_set, assigned_centroid, mode_rank, K, mode_ranking_weight):
    rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")
    centroids = []
    for k in range(K):
        # Get rankigns for centroid k
        cluster_rankings = incomplete_ranking_set[assigned_centroid == k]
        n_rankings_centroid = cluster_rankings.shape[0]
        
        # Form the set of this rankings with the mode rank for completion and obtain normalize weights given mode weight
        cluster_rankings_with_mode = np.concatenate((cluster_rankings, np.array([mode_rank])))
        weights = np.ones(n_rankings_centroid+1)
        weights[-1] = mode_ranking_weight
        weights_normalized = weights/weights.sum()
        
        # Obtain the new centroid via borda aggregation
        new_centroid = rank_algorithm.aggregate(cluster_rankings_with_mode, sample_weight=weights_normalized)
        centroids.append(new_centroid)
    return np.array(centroids)

def random_choice_from_mask(x):
    # Choice from non zero elements
    return skrandom.np_rand_gen.choice(np.flatnonzero(x))

def assign_cluster_balancing(ranking_centroid_distances, ranking_set):
    n_samples = ranking_centroid_distances.shape[0]
    min_by_columns = np.min(ranking_centroid_distances, axis=1)
    is_min = min_by_columns.reshape(min_by_columns.shape[0],1) == ranking_centroid_distances 
    d_assigned = {}
    centroid_assignment = np.zeros(ranking_set.shape[0])
    for i in range(n_samples):
        r_hashable = ranking_set[i].tobytes()
        if r_hashable in d_assigned:
            # previously seen ranking. Assign to the same cluster
            centroid_assignment[i] = d_assigned[r_hashable] 
        else:
            # no previous assignment of the ranking
            centroid_ind = random_choice_from_mask(is_min[i])
            d_assigned[r_hashable] = centroid_ind
            centroid_assignment[i] = centroid_ind
         
    return centroid_assignment.astype(int)  

def assign_cluster(ranking_centroid_distances):
    return np.argmin(ranking_centroid_distances, axis=1) 

def incomplete_rank_cluster_eval(incomplete_ranking_set, rankings_assigned_centroid, centroids):
    a = intra_cluster_mean_distance(incomplete_ranking_set, rankings_assigned_centroid, centroids)
    b = inter_centroid_mean_distance(centroids)
    return b-a/max(b,a)


def distances_from_centroid(rankings, centroid):
    n_rankings = len(rankings)
    centroid_distances = np.zeros(n_rankings)
    for i in range(n_rankings):
        centroid_distances[i] = normalized_kendall_distance(rankings[i], centroid)
    return centroid_distances


def hashable_array(ar):
    return ar.tobytes()


@functools.lru_cache(maxsize=100000)
def array_from_hashable_array(ar_hashable):
    return np.array(np.frombuffer(ar_hashable, dtype=int))


@functools.lru_cache(maxsize=100000)
def normalized_kendall_distance_cached(r1_bytes, r2_bytes):
    r1 = array_from_hashable_array(r1_bytes)
    r2 = array_from_hashable_array(r2_bytes)
    return _generalized_kendall_distance_fast(r1, r2, True)


def normalized_kendall_distance(r1, r2):
    return normalized_kendall_distance_cached(hashable_array(r1), hashable_array(r2))


def random_first_rank_initialization(ranking_set, mode):
    first_centroid_candidate = skrandom.np_rand_gen.choice(ranking_set)
    first_centroid = complete_ranking_with_reference(first_centroid_candidate, mode)
    return first_centroid


def farther_ranking_weight_probability(distances):
    return distances/distances.sum()


def farther_ranking_sample(incomplete_ranking_set, distances):
    probs = farther_ranking_weight_probability(distances)
    return incomplete_ranking_set[skrandom.np_rand_gen.choice(np.arange(len(incomplete_ranking_set)), p=probs)]


def complete_ranking_with_reference(ranking, complete_reference):
    Y = np.array([ranking])
    complete_rankings_farthest(Y, complete_reference)
    return Y[0]


def intra_cluster_mean_distance(incomplete_ranking_set, rankings_assigned_centroid, centroids):
    K = centroids.shape[0]
    K_seen = 0
    a_t = 0
    for k in range(K):
        rankings_in_cluster = incomplete_ranking_set[rankings_assigned_centroid==k]
        if rankings_in_cluster.shape[0] == 0:
            continue
        K_seen += 1  # In case the cluster turns out to be empty after assignment
        distance_sum = 0
        for pi in rankings_in_cluster:
            distance_sum += normalized_kendall_distance(pi, centroids[k])
        ak = distance_sum / rankings_in_cluster.shape[0]
        a_t += ak
    return a_t / K    


def inter_centroid_mean_distance(centroids):
    distance_sum = 0
    n = 0
    for k1 in range(centroids.shape[0]):
        for k2 in range(k1+1, centroids.shape[0]):
            distance_sum += normalized_kendall_distance(centroids[k1], centroids[k2])
            n += 1
    return distance_sum / n


