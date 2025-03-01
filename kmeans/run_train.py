
"""
from sklearn.datasets import load_breast_cancer, make_classification

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import tqdm, copy
import xgboost as xgb
import lightgbm as lgb

import torch

from tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.config import ModelInterfaceConfig

from pmlb import fetch_data, regression_dataset_names
import argparse, os

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

from decoder import PositionalEncoding, TransformerDecoder

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
    # learning rate, patience
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



def main(args):

    pbar = tqdm.tqdm(total=len(regression_dataset_names))
    for ctr, regression_dataset in enumerate(regression_dataset_names):
        break
    X, y = fetch_data(regression_dataset, return_X_y=True)


    # y needs to be processed for multi-class classification
    mapping = {val: idx for idx, val in enumerate(np.unique(y))}
    y = np.array([mapping[val] for val in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    fit_mode = "low_memory"
    # Predict a Centrod, so the dimension of the output is the same as the input
    decoder_n_out = X_train.shape[1]
    # Let's say do we maximum of 3 centroids
    num_queries = 3
    model_path = None
    static_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, _ = initialize_tabpfn_model(
        model_path=model_path,
        which="classifier",
        fit_mode=fit_mode,
        static_seed=static_seed,
    )
    model.cache_trainset_representation = False
    # Decode into K-Centroids
    decoder = TransformerDecoder(model.ninp, model.nhid, decoder_n_out, num_queries, num_layers=2, nhead=4, dropout=0.1)
    #model = CustomSequential(model, decoder)
    model = model.to(device)
    decoder = decoder.to(device)

    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
    aux = torch.zeros(
                X_tensor.size(0),
                device=X_tensor.device,
                dtype=X_tensor.dtype,
            )
    # import pdb; pdb.set_trace()
    # Assume the first three is the one we want to predict as centroids
    y_tensor = torch.as_tensor(X_train[0:3,:], dtype=torch.float32, device=device)

    trainable_params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    
    for step in range(10):
        optimizer.zero_grad()
        output = model._forward(
            x=X_tensor, 
            y=aux,
            only_return_standard_out=True,
            categorical_inds=None,
            single_eval_pos=2,
        )
        output = decoder(output)
        output = output.squeeze(1)
        loss = criterion(output, y_tensor)
        print(f"Step: {step}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
"""
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import tqdm, copy
import xgboost as xgb
import lightgbm as lgb
import re
import torch

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
    # learning rate, patience
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
def parse_centroids(file_path):
        """
        Parses a text file containing centroids enclosed in square brackets and converts them into a single PyTorch tensor.
        
        Args:
            file_path (str): Path to the file containing centroids.
            
        Returns:
            torch.Tensor: A stacked tensor where each row represents a centroid.
        """
        with open(file_path, "r") as file:
            content = file.read()

        # Use regex to extract numbers inside each [ ... ] block
        centroid_strings = re.findall(r"\[([^\]]+)\]", content)

        # Convert extracted numbers into PyTorch tensors (supports multiple centroids)
        centroids = []  # This will store the individual centroid tensors.

        # Process each string extracted by the regex.
        for s in centroid_strings:
            # Split the string into separate number strings using whitespace.
            number_str_list = s.split()
            
            # Convert each string in the list to a float.
            float_values = []
            for num_str in number_str_list:
                float_value = float(num_str)
                float_values.append(float_value)
            
            # Create a PyTorch tensor from the list of float values.
            tensor_value = torch.tensor(float_values, dtype=torch.float32)
            
            # Append the tensor to the centroids list.
            centroids.append(tensor_value)

        # Stack tensors into a single 2D PyTorch tensor
        y_tensor = torch.stack(centroids)

        return y_tensor


def main(args):
    
    dataset_files=["train_dataset_3.csv"]
    centroid_files=["centroids_train_dataset_3.csv"]
    """
    Pipeline by init a list of csv
    dataset_files = sorted(glob.glob(os.path.join("train", "train_dataset_*.csv")))
    centroid_files = sorted(glob.glob(os.path.join("results", "centroids_train_dataset_*.csv")))
    """
    for file_path, file_path_cen in zip(dataset_files, centroid_files):
        print(f"Processing dataset: {file_path} with centroids: {file_path_cen}")
        df = pd.read_csv(file_path)
        X = df.drop(columns=['target']) # Drop the target column from the features
        y = df['target']
        X = X.values
        y = y.values    
        # y needs to be processed for multi-class classification
        mapping = {val: idx for idx, val in enumerate(np.unique(y))}
        y = np.array([mapping[val] for val in y])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        fit_mode = "low_memory"
        # Predict a Centrod, so the dimension of the output is the same as the input
        decoder_n_out = X_train.shape[1]
        # Let's say do we maximum of 3 centroids, let do 2 for now
        num_queries = 2
        print(num_queries)
        model_path = None
        static_seed = 42
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, config, _ = initialize_tabpfn_model(
            model_path=model_path,
            which="classifier",
            fit_mode=fit_mode,
            static_seed=static_seed,
        )
        model.cache_trainset_representation = False
        # Decode into K-Centroids
        decoder = TransformerDecoder(model.ninp, model.nhid, decoder_n_out, num_queries, num_layers=2, nhead=4, dropout=0.1)
        #model = CustomSequential(model, decoder)
        model = model.to(device)
        decoder = decoder.to(device)

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
        aux = torch.zeros(
                    X_tensor.size(0),
                    device=X_tensor.device,
                    dtype=X_tensor.dtype,
                )
        
        print(f"Shape of X_tensor: {X_tensor.shape}")

        # import pdb; pdb.set_trace()
        # Assume the first three is the one we want to predict as centroids
        # y_tensor = torch.as_tensor(X_train[0:3,:], dtype=torch.float32, device=device)
        
        
        y_tensor = parse_centroids(file_path_cen)
    

        """lloyd_centroids = torch.tensor([
            [0.77163462, 0.22836538, 0.91105769, 0.08894231, 0.96794872, 0.03205128, 0.90625, 0.09375, 0.7275641, 0.2724359, 0.61137821, 0.38862179, 0.58012821, 0.41987179, 0.7724359, 0.2275641, 0.59294872, 0.40705128, 0.70352564, 0.29647436, 0.12019231, 0.87980769, 0.92387821, 0.07612179, 0.23397436, 0.76602564, 0.98958333, 0.01041667, 0.14503205, 0.57051282, 0.28445513, 0.9375, 0.0625, 0.9599359, 0.0400641, 0.31169872, 0.68830128, 0.98076923, 0.01923077, 0.67948718, 0.32051282, 0.85817308, 0.14182692, 0.78125, 0.21875, 0.92067308, 0.07932692, 0.6161859, 0.3838141, 0.99198718, 0.00801282, 0.0, 1.0, 0.95592949, 0.04407051, 1.0, 0.0, 0.96634615, 0.03365385, 0.96875, 0.03125, 0.6338141, 0.3661859, 0.87980769, 0.12019231, 0.69791667, 0.30208333, 0.36939103, 0.63060897, 0.5536859, 0.4463141, 1.0, 0.0],
            [1.0, 0.0, 0.94189602, 0.05810398, 0.9587156, 0.0412844, 0.88685015, 0.11314985, 0.59250765, 0.40749235, 0.47859327, 0.52140673, 0.68272171, 0.31727829, 0.78975535, 0.21024465, 0.6383792, 0.3616208, 0.68195719, 0.31804281, 1.0, 0.0, 0.86544343, 0.13455657, 0.38073394, 0.61926606, 1.0, 0.0, 0.0, 1.0, 0.0, 0.95948012, 0.04051988, 0.97782875, 0.02217125, 0.31651376, 0.68348624, 1.0, 0.0, 1.0, 0.0, 0.76834862, 0.23165138, 0.81651376, 0.18348624, 0.96636086, 0.03363914, 0.61850153, 0.38149847, 1.0, 0.0, 0.59021407, 0.40978593, 0.92813456, 0.07186544, 1.0, 0.0, 1.0, 0.0, 0.9441896, 0.0558104, 1.0, 0.0, 1.0, 0.0, 0.56498471, 0.43501529, 0.38914373, 0.61085627, 0.0, 1.0, 0.52217125, 0.47782875]
            ], dtype=torch.float32, device=device)
            """

        
        print(f"Shape of y_tensor: {y_tensor.shape}")

        trainable_params = list(model.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss()
        
        for step in range(10):
            optimizer.zero_grad()
            output = model._forward(
                x=X_tensor, 
                y=aux,
                only_return_standard_out=True,
                categorical_inds=None,
                single_eval_pos=2,
            )
            print(f"Shape of model output: {output.shape}")
            output = decoder(output)
            print(f"Shape of output after decoder: {output.shape}")

            output = output.squeeze(1)
            print(f"Shape of output after squeeze: {output.shape}")

            loss = criterion(output, y_tensor)
            print(f"Shape of loss: {loss.shape}")
            print(f"Step: {step}, Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        break #break after 1 dataset


    

if __name__ == "__main__":
    
    args = parse_arguments()
    main(args)