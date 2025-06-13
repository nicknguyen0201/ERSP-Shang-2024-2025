from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
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
from torch.utils.data import TensorDataset, DataLoader
import glob
from plot_centroids import plot_centroids

import wandb
os.makedirs("kumiko_train_plots", exist_ok=True)
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



def fixed_order_centroid_loss(predicted, target):
    """
    Computes MSE loss after sorting both predicted and target centroids by their distance from the origin.
    """
    # Compute norms
    pred_norms = torch.norm(predicted, dim=1)
    targ_norms = torch.norm(target, dim=1)

    # Sort both by norm
    pred_sorted = predicted[torch.argsort(pred_norms)]
    target_sorted = target[torch.argsort(targ_norms)]

    # Mean squared error between sorted
    base_loss = F.mse_loss(pred_sorted, target_sorted)

    # Optional: penalty to avoid centroids collapsing too close
    penalty_weight = 1e-9
    pairwise_dists = pred_sorted.unsqueeze(1) - pred_sorted.unsqueeze(0)
    norms = torch.norm(pairwise_dists, dim=2)
    norms_upper_half = torch.triu(norms, diagonal=1)
    penalty = torch.sum(1 / (norms_upper_half + 1e-6))

    loss = base_loss + penalty_weight * penalty

    # Debugging output
    print(f"[Fixed Order] Base Loss: {base_loss.item()}, Penalty: {penalty_weight * penalty.item()}")
    
    return loss
    
    
def hungarian_centroid_loss(predicted, target):
    """
    Computes permutation-invariant Euclidean norm loss between predicted and target centroids
    using the Hungarian algorithm.

    Args:
        predicted: Tensor of shape [k, D] - model output centroids
        target: Tensor of shape [k, D] - true centroids

    Returns:
        Euclidean norm loss after optimal assignment
    """
    k = predicted.shape[0]
    cost_matrix = torch.cdist(predicted, target, p=2).cpu().detach().numpy()  # [k, k] Euclidean distances

    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Optimal assignment

    # Convert to torch tensors for indexing
    predicted = predicted[row_ind]
    target = target[col_ind]

    # variables for penalty
    penalty_weight = 1e-9

    # Compute base Euclidean norm loss
    #base_loss = torch.norm(predicted - target, p=2)  # L2 norm
    base_loss = F.mse_loss(predicted, target)  # Mean Squared Error
    
    
    # distance between each predicted centroid
    pairwise_dists = predicted.unsqueeze(1) - predicted.unsqueeze(0)

    # compute norm
    norms = torch.norm(pairwise_dists, dim=2)
    norms_upper_half = torch.triu(norms, diagonal=1)

    # calculate penalty (inverse distance penalty)
    penalty = torch.sum(1 / (norms_upper_half + 1e-6))

    # apply penalty
    loss = base_loss + penalty_weight * penalty

    # debugging
    print(f"Base Loss: {base_loss.item()}, Penalty: {penalty_weight * penalty.item()}")
    

    return loss

def main(args):
    # Initialize wandb with a project name and configuration
    wandb.init(project="ERSP_transformer", config={
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": 150,
        "batch_size_fraction": 0.4  # Using 40% of dataset per batch the code
    })
    # path to current model state_dict
    model_state_path = "./new_data_model_3.pth"

   
   
    
    
    dataset_files = sorted(glob.glob("new_data_3/train_no_header/*.csv"))
    centroid_files = sorted(glob.glob("new_data_3/train_result/*.csv"))

    #file_path_cv="new_data/cv/cv_no_header/cv_dataset_46876.csv"

    #file_path_cen_cv= "new_data/cv/cv_result/centroids_cv_dataset_46876.csv"
    # Set device (use CUDA if available)
    if len(dataset_files) != len(centroid_files):
        raise ValueError("Mismatch between dataset files and centroid files.")

    print(f"Using {len(dataset_files)} datasets for training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Determining maximum centroid dimensionality...")
    max_centroid_dim = 0
    for file_path_cen in centroid_files:
        y_tensor = parse_centroids(file_path_cen, device)
        if y_tensor.shape[1] > max_centroid_dim:
            max_centroid_dim = y_tensor.shape[1]
    print(f" Max centroid dimension found: {max_centroid_dim}")

    count_datasets=1
    num_of_files = len(dataset_files)
    num_of_centroids_file = len(centroid_files)
    if num_of_files>0 and num_of_centroids_file==num_of_files:
        for file_path, file_path_cen in zip(dataset_files, centroid_files):
            print(f"Processing dataset: {file_path} with centroids: {file_path_cen}")
            print(f"Dataset #{count_datasets} / {num_of_files}")
            count_datasets+=1
           
            fit_mode = "low_memory"
            #decoder_n_out = X_train.shape[1]  # Output dimension should match input features
            df = pd.read_csv(file_path)
            X=df.values

            """ run cv
            df = pd.read_csv(file_path_cv)
            X_cv=df.values
            """

            decoder_n_out = max_centroid_dim
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
            decoder = TransformerDecoder(model.ninp, model.nhid, decoder_n_out, num_queries, num_layers=5, nhead=4, dropout=0.1, sigma=0.0)
            #dropout used to be 0.1, but I will let it =0.0 for better overfitting

            # Move model and decoder to device
            model = model.to(device)
            decoder = decoder.to(device)

            # Move data tensors to the same device
            #X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)

            #cv tensor
            #X_cv_tensor = torch.as_tensor(X_cv, dtype=torch.float32, device=device).unsqueeze(1)
            
            
           
            aux = torch.zeros(X_tensor.size(0), dtype=X_tensor.dtype, device=device)

            #aux_cv = torch.zeros(X_cv_tensor.size(0), dtype=X_cv_tensor.dtype, device=device)

            #aux = -1 * torch.ones(X_tensor.shape[0], dtype=torch.float32, device=device)
            # print(f"Shape of X_tensor: {X_tensor.shape}")

            # Parse centroids and move to the same device
            y_tensor = parse_centroids(file_path_cen, device)
            orig_centroid_dim = y_tensor.shape[1]

            if orig_centroid_dim < max_centroid_dim:
                pad_size = max_centroid_dim - orig_centroid_dim
                padding = torch.zeros((y_tensor.shape[0], pad_size), device=device)
                y_tensor = torch.cat([y_tensor, padding], dim=1)

            #y_cv_tensor = parse_centroids(file_path_cen_cv, device)
            
            # Create a Dataset and DataLoader for mini-batch training
            #dataset = TensorDataset(X_tensor, aux)
            #batch_size = max(1, int(0.4 * len(dataset)))  # Use 40% of the dataset per batch (ensuring at least 1 sample)
            #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Define optimizer and loss function
            trainable_params = list(model.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            #criterion = torch.nn.MSELoss().to(device)  # Move loss function to the correct device


            # Training loop (epochs)
            for step in range(13):
                # --- Training Phase ---
                model.train()  # Set model to training mode
                #for batch in data_loader:
                optimizer.zero_grad()
                #batch_X, batch_aux = batch
                output = model._forward(
                    x=X_tensor, 
                    y=aux,
                    only_return_standard_out=True,
                    categorical_inds=None,
                    single_eval_pos=2,
                )
                output = decoder(output)
                output = output.squeeze(1)                
                # Move output to the same device as y_tensor before computing loss
                output = output.to(device)               
                loss = fixed_order_centroid_loss(output, y_tensor)
                
                print(f"Step: {step}, Loss: {loss.item()}")

                loss.backward()
                optimizer.step()
                

               
        
                """
                # --- Validation Phase ---
                model.eval()  # Switch model to evaluation mode
                with torch.no_grad():
                    val_output = model._forward(
                        x=X_cv_tensor,  # Validation features tensor (20%)
                        y=aux_cv,       # Validation auxiliary tensor if needed
                        only_return_standard_out=True,
                        categorical_inds=None,
                        single_eval_pos=2,
                    )
                    val_output = decoder(val_output)
                    val_output = val_output.squeeze(1)
                    val_output = val_output.to(device)
                    
                    # Compute validation loss
                    val_loss = fixed_order_centroid_loss(val_output, y_cv_tensor)
                
                # Log the validation loss for this step to wandb
                wandb.log({
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "epoch": step
                })
                print(f"Step: {step}, Val Loss: {val_loss.item()}")
                """
                
                
                            

        


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
    
    

    



        


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
