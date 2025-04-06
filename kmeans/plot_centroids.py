import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
"""
def plot_centroids(data_points, predicted_centroids, ground_truth_centroids, save_path=None):
    # Convert to numpy arrays for easy manipulation
    data_points = data_points.cpu().numpy() if isinstance(data_points, torch.Tensor) else data_points
    predicted_centroids = predicted_centroids.cpu().numpy() if isinstance(predicted_centroids, torch.Tensor) else predicted_centroids
    ground_truth_centroids = ground_truth_centroids.cpu().numpy() if isinstance(ground_truth_centroids, torch.Tensor) else ground_truth_centroids

    # Print shape of centroids to understand their dimensions
    print("Predicted Centroids shape:", predicted_centroids.shape)
    print("Ground Truth Centroids shape:", ground_truth_centroids.shape)

    # Plot data points
    plt.scatter(data_points[:, 0], data_points[:, 1], color='gray', alpha=0.5, label='Data Points')

    # Plot predicted centroids (now reduced to 2D if needed)
    plt.scatter(predicted_centroids[0], predicted_centroids[1], color='red', marker='x', label='Predicted Centroids')

    # Plot ground truth centroids (now reduced to 2D if needed)
    plt.scatter(ground_truth_centroids[0], ground_truth_centroids[1], color='blue', marker='o', label='Ground Truth Centroids')

    # Add labels and legend
    plt.title("Data Points with Predicted and Ground Truth Centroids (Reduced to 2D if needed)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    # Save the plot as a PDF if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf')  # Save the plot as a PDF

    # Show plot
    plt.show()"""
# we forgot to use PCA
def plot_centroids(data_points, predicted_centroids, ground_truth_centroids, save_path=None):
    # Convert to numpy arrays if needed
    if torch.is_tensor(data_points):
        data_points = data_points.cpu().numpy()
    if torch.is_tensor(predicted_centroids):
        predicted_centroids = predicted_centroids.cpu().numpy()
    if torch.is_tensor(ground_truth_centroids):
        ground_truth_centroids = ground_truth_centroids.cpu().numpy()

    # Print shapes for debugging
    print("Data Points shape:", data_points.shape)
    print("Predicted Centroids shape:", predicted_centroids.shape)
    print("Ground Truth Centroids shape:", ground_truth_centroids.shape)

    # If data is higher than 2D, apply PCA to reduce to 2 components
    if data_points.shape[1] > 2:
        pca = PCA(n_components=2)
        data_points = pca.fit_transform(data_points)
        predicted_centroids = pca.transform(predicted_centroids)
        ground_truth_centroids = pca.transform(ground_truth_centroids)
        print("Applied PCA to reduce dimensions to 2.")

    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(data_points[:, 0], data_points[:, 1],
                color='gray', alpha=0.5, label='Data Points')

    # Plot predicted centroids (assumed to be of shape (n_centroids, 2))
    plt.scatter(predicted_centroids[:, 0], predicted_centroids[:, 1],
                color='red', marker='x', s=100, label='Predicted Centroids')

    # Plot ground truth centroids (assumed to be of shape (n_centroids, 2))
    plt.scatter(ground_truth_centroids[:, 0], ground_truth_centroids[:, 1],
                color='blue', marker='o', s=100, label='Ground Truth Centroids')

    # Add title, labels, legend, and grid
    plt.title("Data Points with Predicted and Ground Truth Centroids (Projected to 2D)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"Plot saved to {save_path}")

    plt.show()
