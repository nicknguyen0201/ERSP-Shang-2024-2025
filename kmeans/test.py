import argparse
import glob
import os
import re
import torch
import pandas as pd
import shutil
from torch.utils.data import DataLoader, TensorDataset
from tabpfn.base import initialize_tabpfn_model
from decoder import TransformerDecoder
from plot_centroids import plot_centroids
from fixed_order_centroid import parse_centroids, fixed_order_centroid_loss, visualize_centroids
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
import time  # Import the time module


def assign_clusters_knn(X, centroids):
    # X: (N, D), centroids: (K, D)
    dists = torch.cdist(X, centroids, p=2)  # shape: (N, K)
    return torch.argmin(dists, dim=1)  # cluster indices for each point


# Clustering accuracy using Hungarian matching on confusion matrix
def clustering_accuracy(true_labels, predicted_labels):
    y_true = true_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()
    C = confusion_matrix(y_true, y_pred)

    row_ind, col_ind = linear_sum_assignment(-C)  # We use negative because we want to maximize
    total_correct = C[row_ind, col_ind].sum()

    accuracy = total_correct / np.sum(C)
    return accuracy

# Load CSV as float tensor
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    return torch.tensor(df.values, dtype=torch.float32)


# Clear a folder and its contents
def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
    else:
        os.makedirs(folder_path)

# Run model and decoder to predict centroids
def test_model(X_test, model, decoder, device):
    model.eval()
    decoder.eval()

    X_test_tensor = X_test.to(device).unsqueeze(1)
    aux_test = torch.zeros(X_test_tensor.shape[0], dtype=torch.float32, device=device)

    # Start the timer
    start_time = time.perf_counter()

    with torch.no_grad():
        transformer_output = model._forward(
            x=X_test_tensor,
            y=aux_test,
            only_return_standard_out=True,
            categorical_inds=None,
            single_eval_pos=2,
        )
        pred = decoder(transformer_output).squeeze(1)

    # End the timer
    end_time = time.perf_counter()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Prediction time: {elapsed_time:.4f} seconds") 

    return pred


def load_true_cluster_ids(file_path):
    """
    Load true cluster IDs from a file and return as a PyTorch tensor.
    Args:
        file_path (str): Path to the file containing true cluster IDs.
    Returns:
        torch.Tensor: Tensor containing true cluster IDs.
    """

    true_labels = pd.read_csv(file_path, header=None).squeeze("columns")
    true_cluster_ids = torch.tensor(true_labels.values, dtype=torch.long)
    return true_cluster_ids


# Evaluate model predictions over all test sets
def process_folder(test_folder, ground_truth_folder, exact_true_labels,model, decoder_state_dict, device, args):
    test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_folder, "*.csv")))
    exact_true_labels_files = sorted(glob.glob(os.path.join(exact_true_labels, "*.csv")))

    output_folder = "new_data_output"
    clear_output_folder(output_folder)

    print("Determining max centroid dimensionality from ground truth files...")
    max_centroid_dim = 0
    for gt_file in ground_truth_files:
        centroids = parse_centroids(gt_file, device)
        if centroids.shape[1] > max_centroid_dim:
            max_centroid_dim = centroids.shape[1]
    print(f"Max test-time centroid dimension: {max_centroid_dim}")

    trained_decoder_out_dim = 20
    print(f"Using decoder output dim: {trained_decoder_out_dim} (from training)")

    if len(test_files) != len(ground_truth_files):
        print("Warning: Mismatch in number of test and ground truth files.")
        return

    for test_file, ground_truth_file, exact_true_labels_file in zip(test_files, ground_truth_files,exact_true_labels_files):
        print(f"\nProcessing {test_file}...")
        print(f"Exact_true_labels_file {exact_true_labels_file}...")

        # Load test data and ground truth
        X_test = load_data(test_file)
       
        ground_truth_centroids = parse_centroids(ground_truth_file, device)
        true_dim = ground_truth_centroids.shape[1]

        # Define decoder using the *trained* output dim
        decoder = TransformerDecoder(
            model.ninp, model.nhid, trained_decoder_out_dim, args.num_queries,
            num_layers=5, nhead=4, dropout=0.0
        ).to(device)

        # Load full state dict
        decoder.load_state_dict(decoder_state_dict)

        # Predict centroids and slice down to match ground truth dim
        print("X_test shape:", X_test.shape)
        predictions = test_model(X_test, model, decoder, device)
        print("Predictions shape before slicing:", predictions.shape)
        predictions = predictions[:, :true_dim]
        print("Predictions shape after slicing:", predictions.shape)
        print("Evaluating with fixed-order loss...")
        loss = fixed_order_centroid_loss(predictions, ground_truth_centroids)
        print(f"Fixed-order evaluation loss: {loss.item()}")

        # Centroid clustering metrics  
        pred_cluster_ids = assign_clusters_knn(X_test.to(device), predictions)
        true_cluster_ids = load_true_cluster_ids(exact_true_labels_file)
        acc = clustering_accuracy(true_cluster_ids, pred_cluster_ids)
        print(f"Clustering Accuracy w original dataset (Hungarian Matched): {acc:.4f}")

        true_cluster_ids = assign_clusters_knn(X_test.to(device), ground_truth_centroids)
        acc = clustering_accuracy(true_cluster_ids, pred_cluster_ids)
        print(f"Clustering Accuracy (Hungarian Matched): {acc:.4f}")
      
        
        # Save original centroid plot
        save_path = os.path.join(output_folder, os.path.basename(test_file).replace(".csv", ".pdf"))
        plot_centroids(X_test, predictions.cpu().numpy(), ground_truth_centroids.cpu().numpy(), save_path)

        # Sorted plot
        sorted_plot_dir = os.path.join(output_folder, "sorted_plots")
        os.makedirs(sorted_plot_dir, exist_ok=True)
        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        sorted_save_path = os.path.join(sorted_plot_dir, f"sorted_{test_file_name}.pdf")

        pred_sorted = predictions[torch.argsort(torch.norm(predictions, dim=1))].detach().cpu().numpy()
        gt_sorted = ground_truth_centroids[torch.argsort(torch.norm(ground_truth_centroids, dim=1))].detach().cpu().numpy()
        plot_centroids(X_test, pred_sorted, gt_sorted, sorted_save_path)


def main(args):
    model, config, _ = initialize_tabpfn_model(
        model_path=None,
        which="classifier",
        fit_mode=args.fit_mode,
        static_seed=args.static_seed
    )

    checkpoint = torch.load("new_data_model.pth", map_location=args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder_state_dict = checkpoint['decoder_state_dict']

    model = model.to(args.device)

    process_folder(
        args.test_folder,
        args.ground_truth_folder,
        "new_data/test_w_exact_true_labels",
        model,
        decoder_state_dict,
        args.device,
        args
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth centroids.")
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--ground_truth_folder", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fit_mode", type=str, default="low_memory")
    parser.add_argument("--static_seed", type=int, default=42)
    parser.add_argument("--num_queries", type=int, default=2)
    main(parser.parse_args())