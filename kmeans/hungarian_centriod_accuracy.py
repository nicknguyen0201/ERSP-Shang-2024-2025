import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
#pass in both both ground truth and predicted centriods as a ndarray K x d
def centroid_matching_accuracy(ground_truth_centroids, predicted_centroids):
    if isinstance(ground_truth_centroids, torch.Tensor):
        ground_truth_centroids = ground_truth_centroids.cpu().numpy()
    if isinstance(predicted_centroids, torch.Tensor):
        predicted_centroids = predicted_centroids.cpu().numpy()

    # Compute the cost matrix
    
    cost_matrix = cdist(ground_truth_centroids, predicted_centroids, metric='sqeuclidean')
    print('Datatype:', ground_truth_centroids.dtype)
    print('Datatype:', predicted_centroids.dtype)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mse = np.mean([cost_matrix[i, j] for i, j in zip(row_ind, col_ind)])
 
    return mse
