# Makes Distance Matrices

import numpy as np
from distances import euclidean_distance, manhattan_distance, cosine_similarity, minkowski_distance, gower_distance
from functools import partial

metrics = {
    'euclidean' : euclidean_distance,
    'manhattan' : manhattan_distance,
    'cosine'    : cosine_similarity,
    'minkowski' : minkowski_distance,
    "gower"     : gower_distance
}

def make_distance_matrix(data, metric='gower') -> np.ndarray:
    """
    Constructs a distance matrix for a given dataset `data` using the specified distance `metric`. 
    Input:
    - data: A 2D numpy array where each element represents a  point.
    - metric: A string specifying the distance metric to use. Options are 'euclidean', 'manhattan', 'cosine', 'minkowski', and 'gower'. Gower distance is chosen by default. Note that gower distance returns manhattan distance for purely numerical data.
    Output:
    - A 2D numpy array representing the distance matrix.
    """
    dim = data.shape[0]

    if metric not in metrics:
        raise ValueError(f"Invalid distance metric: {metric}")
    elif metric == 'gower':


        def compute_ranges(data):
            num_features = data.shape[1]
            ranges = np.zeros(num_features)

            for i in range(num_features):
                col = data[:, i]
                numeric_mask = np.vectorize(lambda x: isinstance(x, (int,float)))(col)
                numeric_values = col[numeric_mask]
                ranges[i] = numeric_values.max() - numeric_values.min() if numeric_values.size > 0 else 0
            return ranges

        
        data_range = compute_ranges(data)
        func = partial(gower_distance, ranges=data_range)

    else:
        func = metrics[metric]

    matrix = np.zeros((dim, dim))
        
    for x in range(dim):
        for y in range(x, dim):
            if x == y:
                matrix[x][y] = 0
            else:
                d = func(data[x], data[y])
                matrix[x][y] = d
                matrix[y][x] = d

    return matrix
