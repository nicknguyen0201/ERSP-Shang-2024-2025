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
from hungarian_centriod_accuracy import centroid_matching_accuracy
from plot_centroids import plot_centroids


def load_data(file_path):
    """Loads data from a CSV file and returns a tensor."""
    df = pd.read_csv(file_path)
    return torch.tensor(df.values, dtype=torch.float32)


def parse_centroids(file_path, device):
    """Parses a file containing centroids and returns a PyTorch tensor."""
    with open(file_path, "r") as file:
        content = file.read()
    
    centroid_strings = re.findall(r"\[([^\]]+)\]", content)
    centroids = [torch.tensor([float(num) for num in s.split()], dtype=torch.float32).to(device) for s in centroid_strings]
    
    return torch.stack(centroids).to(device) if centroids else torch.empty(0, device=device)

def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
    else:
        os.makedirs(folder_path)


def test_model(X_test, model, decoder, device, file_path_cen):
    """Tests the model and returns predictions."""
    model.eval()
    decoder.eval()

    X_test_tensor = X_test.to(device).unsqueeze(1)

    # Set aux_test to -1 for all entries (indicating unknown labels to predict)
    #aux_test = -1 * torch.ones(X_test_tensor.shape[0], dtype=torch.float32, device=device)

    aux_test = torch.full((X_test_tensor.shape[0],), float('nan'), dtype=torch.float32, device=device)

    with torch.no_grad():
        transformer_output = model._forward(
            x=X_test_tensor,
            y=aux_test,
            only_return_standard_out=True,
            categorical_inds=None,
            single_eval_pos=2,
        )
        pred = decoder(transformer_output).squeeze(1)
        print(pred.shape)
    
    return pred.cpu().numpy()


def evaluate(predictions, ground_truth_file, device):
    """Evaluates predictions using centroid matching accuracy."""
    y_true = parse_centroids(ground_truth_file, device)
    print(f"Evaluating... Ground truth shape: {y_true.shape}, Predictions shape: {predictions.shape}")
    accuracy = centroid_matching_accuracy(y_true, predictions)
    print(f"Hungarian Centroid Accuracy: {accuracy}")
    return accuracy


def process_folder(test_folder, ground_truth_folder, model, device, args):
    """Processes each test file and its corresponding ground truth."""
    test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_folder, "*.csv")))
    
    if len(test_files) != len(ground_truth_files):
        print("Warning: The number of test files and ground truth files do not match!")
        return

    output_folder = "output"
    clear_output_folder(output_folder)

    for test_file, ground_truth_file in zip(test_files, ground_truth_files):
        print(f"Processing {test_file} and ground truth {ground_truth_file}...")
        X_test = load_data(test_file)
        decoder_n_out = X_test.shape[1]

        decoder = TransformerDecoder(
            model.ninp, model.nhid, decoder_n_out, args.num_queries,
            num_layers=5, nhead=4, dropout=0.0
        ).to(device)

        predictions = test_model(X_test, model, decoder, device, ground_truth_file)


        ground_truth_centroids = parse_centroids(ground_truth_file, device)

        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        save_path = f"output/{test_file_name}.pdf"
        plot_centroids(X_test, predictions, ground_truth_centroids, save_path)

        evaluate(predictions, ground_truth_file, device)  


def main(args):
    """Initializes the model and processes the test folder."""
    model, config, _ = initialize_tabpfn_model(
        model_path=None,
        which="classifier",
        fit_mode=args.fit_mode,
        static_seed=args.static_seed
    )

    checkpoint = torch.load("kumiko_graph_test.pth", map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("'optimizer_state_dict' not found in checkpoint")
    """
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("'model_state_dict' not found in checkpoint")   
    if 'decoder_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        raise KeyError("'decoder_state_dict' not found in checkpoint")
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        raise KeyError("'optimizer_state_dict' not found in checkpoint")"""
    
    model = model.to(args.device)
    process_folder(args.test_folder, args.ground_truth_folder, model, args.device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth.")
    parser.add_argument("--test_folder", type=str, required=True, help="Path to test CSV files")
    parser.add_argument("--ground_truth_folder", type=str, required=True, help="Path to ground truth CSV files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--fit_mode", type=str, default="low_memory", help="Fit mode for the model")
    parser.add_argument("--static_seed", type=int, default=42, help="Static seed for reproducibility")
    parser.add_argument("--num_queries", type=int, default=2, help="Number of queries for the decoder")
    
    main(parser.parse_args())