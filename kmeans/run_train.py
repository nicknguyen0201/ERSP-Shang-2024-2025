from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from hungarian_centriod_accuracy import centroid_matching_accuracy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import tqdm, copy
import xgboost as xgb
import lightgbm as lgb
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.config import ModelInterfaceConfig

from pmlb import fetch_data, regression_dataset_names
import argparse, os
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from decoder import PositionalEncoding, TransformerDecoder

import glob

def get_trainable_parameters(obj):
    trainable_params = []

    def recursive_search(module):
        if isinstance(module, nn.Module):
            trainable_params.extend([p for p in module.parameters() if p.requires_grad])
        elif isinstance(module, dict):  # Search dictionaries
            for v in module.values():
                recursive_search(v)
        elif isinstance(module, (list, tuple, set)):  # Search lists, tuples, sets
            for v in module:
                recursive_search(v)
        elif hasattr(module, "__dict__"):  # Search objects with attributes
            for attr_name, attr_value in vars(module).items():
                recursive_search(attr_value)

    recursive_search(obj)
    return trainable_params


def parse_arguments():
    # Learning rate, patience
    parser = argparse.ArgumentParser(description="K-Centroids")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the model"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the model"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping"
    )

    return parser.parse_args()


def parse_centroids(file_path, device):
    """
    Parses a text file containing centroids enclosed in square brackets and converts them into a single PyTorch tensor.
    
    Args:
        file_path (str): Path to the file containing centroids.
        device (torch.device): The device to move tensors to.

    Returns:
        torch.Tensor: A stacked tensor where each row represents a centroid.
    """
    with open(file_path, "r") as file:
        content = file.read()

    centroid_strings = re.findall(r"\[([^\]]+)\]", content)
    centroids = []

    for s in centroid_strings:
        float_values = [float(num_str) for num_str in s.split()]
        tensor_value = torch.tensor(float_values, dtype=torch.float32).to(device)  # Move to correct device
        centroids.append(tensor_value)

    y_tensor = torch.stack(centroids).to(device)  # Ensure stacked tensor is on correct device
    return y_tensor



def hungarian_centroid_loss(predicted, target):
    """
    Computes permutation-invariant MSE loss between predicted and target centroids
    using the Hungarian algorithm.

    Args:
        predicted: Tensor of shape [k, D] - model output centroids
        target: Tensor of shape [k, D] - true centroids

    Returns:
        Mean squared loss after optimal assignment
    """
    k = predicted.shape[0]
    cost_matrix = torch.cdist(predicted, target, p=2).cpu().detach().numpy()  # [k, k] Euclidean distances

    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Optimal assignment

    # Convert to torch tensors for indexing
    predicted = predicted[row_ind]
    target = target[col_ind]

    # Compute MSE between matched pairs
    return F.mse_loss(predicted, target)



def plot_dataset_clusters(X, true_centroids, pred_centroids, dataset_name, device='cpu'):
    """
    [input argument descriptions]
        X = input data tensor (N x D)
        true_centroids = ground truth centroids tensor (K x D)
        pred_centroids = predicted centroids tensor (K x D)
        dataset_name = name for plot title
        device = device where tensors reside
    """
    # Convert tensors to numpy arrays on CPU
    X_np = X.cpu().numpy().squeeze()
    true_np = true_centroids.cpu().numpy()
    pred_np = pred_centroids.cpu().numpy()

    plt.figure(figsize=(10, 6))
    
    # data points
    plt.scatter(X_np[:, 0], X_np[:, 1], c='gray', alpha=0.4, label='Data Points')
    
    # ground truth centroids
    plt.scatter(true_np[:, 0], true_np[:, 1], marker='*', s=400, c='red', edgecolor='black', linewidth=2, label='Ground Truth Centroids')
    
    # predicted centroids (with Hungarian matching?)
    cost_matrix = np.linalg.norm(true_np[:, np.newaxis] - pred_np, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    plt.scatter(pred_np[:, 0], pred_np[:, 1], marker='X', s=200, c='green', edgecolor='black', linewidth=2, label='Predicted Centroids')
    
    # draw lines between matched centroids
    for i, j in zip(row_ind, col_ind):
        plt.plot([true_np[i, 0], pred_np[j, 0]], [true_np[i, 1], pred_np[j, 1]], 'k--', linewidth=1, alpha=0.7)

    plt.title(f'Clustering Results: {dataset_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    # save the plot
    plt.savefig("./training_centroids/" + dataset_name+".pdf", dpi=300, bbox_inches='tight')
    print(f"Plot saved to ./training_centroids/{dataset_name}")



def main(args):

    # path to current model state_dict
    model_state_path = "./kumiko_graph_test.pth"

    # dataset files for training
    # dataset_files = ["train_dataset_3.csv"]
    # centroid_files = ["centroids_train_dataset_3.csv"]
    dataset_files = sorted(glob.glob("sub_sampling_train(nick)/*.csv"))
    centroid_files = sorted(glob.glob("sub_sampling_result(nick)/*.csv"))

    # Set device (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    count_datasets=1
    num_of_files = len(dataset_files)
    num_of_centroids_file = len(centroid_files)
    if num_of_files>0 and num_of_centroids_file==num_of_files:
        for file_path, file_path_cen in zip(dataset_files, centroid_files):
            print(f"Processing dataset: {file_path} with centroids: {file_path_cen}")
            print(f"Dataset #{count_datasets} / {num_of_files}")
            count_datasets+=1
            # df = pd.read_csv(file_path, header = None)
            # if df.shape[1] > 1:
            #     X = df.iloc[:, :-1].values  # features
            #     y = df.iloc[:, -1].values   # target (last column)
            # else:
            #     X = df.values
            #     y = None  # or np.zeros(len(df)) if needed
            """df = pd.read_csv(file_path)
            X = df.drop(columns=['target']).values  # Drop the target column from the features
            y = df['target'].values    
            # Check for NaNs in the target column
            if pd.isnull(y).any():
                print(f"Warning: NaN values found in target column of {file_path}. Dropping rows with NaNs.")
                df = df.dropna(subset=['target'])
                X = df.drop(columns=['target']).values
                y = df['target'].values
            else:
                X = df.drop(columns=['target']).values
            """
            # Convert target labels to numerical values
            #mapping = {val: idx for idx, val in enumerate(np.unique(y))}
            #y = np.array([mapping[val] for val in y])

            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = None, random_state=None)
            fit_mode = "low_memory"
            #decoder_n_out = X_train.shape[1]  # Output dimension should match input features
            df = pd.read_csv(file_path)
            X=df.values
            decoder_n_out = X.shape[1]
            num_queries = 2  # Number of centroids
            model_path = None
            static_seed = 42

            # Initialize TabPFN model
            model, config, _ = initialize_tabpfn_model(
                model_path=model_path,
                which="classifier",
                fit_mode=fit_mode,
                static_seed=static_seed,
            )
            model.cache_trainset_representation = False

            # Initialize Transformer Decoder
            decoder = TransformerDecoder(model.ninp, model.nhid, decoder_n_out, num_queries, num_layers=5, nhead=4, dropout=0.1, sigma=0.1)
            #dropout used to be 0.1, but I will let it =0.0 for better overfitting

            # Move model and decoder to device
            model = model.to(device)
            decoder = decoder.to(device)

            # Move data tensors to the same device
            #X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
            aux = torch.zeros(X_tensor.size(0), dtype=X_tensor.dtype, device=device)

            # print(f"Shape of X_tensor: {X_tensor.shape}")

            # Parse centroids and move to the same device
            y_tensor = parse_centroids(file_path_cen, device)

            
            # Define optimizer and loss function
            trainable_params = list(model.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            criterion = torch.nn.MSELoss().to(device)  # Move loss function to the correct device

            # Debugging: Print tensor devices
            print(f"Device of model: {next(model.parameters()).device}")
            print(f"Device of decoder: {next(decoder.parameters()).device}")
            print(f"Device of X_tensor: {X_tensor.device}")
            print(f"Device of y_tensor: {y_tensor.device}")

            # Training loop (epochs)
            for step in range(150):# I changed from 1000 to 200
                optimizer.zero_grad()

                output = model._forward(
                    x=X_tensor, 
                    y=aux,
                    only_return_standard_out=True,
                    categorical_inds=None,
                    single_eval_pos=2,
                )

                # print(f"Shape of model output: {output.shape}")

                output = decoder(output)
                """print(f"Shape of output after decoder: {output.shape}")
                for i, centroid in enumerate(output):
                    print(f"Centroid {i+1}: {centroid}")"""

                output = output.squeeze(1)
                """print(f"Shape of output after squeeze: {output.shape}")
                for i, centroid in enumerate(output):
                    print(f"Centroid {i+1}: {centroid}")"""

                # Move output to the same device as y_tensor before computing loss
                output = output.to(device)
                """print(f"Shape of y_tensor before we go into loss function with output: {y_tensor.shape}")
                for i, centroid in enumerate(y_tensor):
                    print(f"Centroid {i+1}: {centroid}")"""
                #print(f"Decoder output shape: {output.shape}")
                #print(f"Ground truth y_tensor shape: {y_tensor.shape}")

                loss = hungarian_centroid_loss(output, y_tensor)
                print(f"Step: {step}, Loss: {loss.item()}")

                loss.backward()
                optimizer.step()

            # After training loop finishes:
            #print("Evaluating predicted vs. true centroids")
            #mse_loss, centroids_pred_np, true_centroids_np = test_transformer(model, decoder, X_test, y_tensor, device=device)

            # for plotting
            with torch.no_grad():
                output = model._forward(X_tensor, aux, only_return_standard_out=True, categorical_inds=None, single_eval_pos=2)
                pred_centroids = decoder(output).squeeze(1)
            
                # Plot results for this dataset
                plot_dataset_clusters(X_tensor, y_tensor, pred_centroids, os.path.basename(file_path), device)

    
        print("No saved model found. Starting from scratch.")
        # Save the model state_dict

        torch.save({
            'model_state_dict': model.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_state_path)
        
        print("Model saved to '" + model_state_path + "'")
    else:
        print("nothing runs")
    
    '''
        # Load the model state_dict
        
        if os.path.exists(model_state_path):
            print("Loading model from '" + model_state_path + "'")
            checkpoint = torch.load(model_state_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
        '''

    



        


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
