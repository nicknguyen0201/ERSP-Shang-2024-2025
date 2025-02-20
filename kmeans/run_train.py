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