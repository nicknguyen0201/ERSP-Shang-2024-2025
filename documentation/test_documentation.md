# Documentation: `run_train.py`

This script evaluates a trained TabPFN + Transformer decoder model on test datasets by predicting centroids and comparing them with ground truth centroids using fixed-order loss and clustering metrics. It is part of a research project exploring Transformer-based approximation for NP-Hard problems such as k-means clustering.

---

## Overview

- **Goal:** Evaluate predicted k-means centroids from a trained model and compute clustering accuracy.
- **Libraries:** `torch`, `pandas`, `tabpfn`, `sklearn`, `scipy.optimize`, and custom modules (`fixed_order_centroid`, `plot_centroids`, etc.).
- **Input:**
  - Tabular test datasets (`.csv`)
  - Ground truth centroids (from ELKI, `.txt`)
  - Ground truth cluster labels (`.csv`)
- **Output:**
  - Evaluation metrics (accuracy, loss)
  - Cluster visualizations (PDFs of centroid plots)

---

## Main Components

### 1. `assign_clusters_knn(X, centroids)`

Assigns each data point in `X` to its nearest predicted centroid using L2 (Euclidean) distance.

### 2. `clustering_accuracy(true_labels, predicted_labels)`

Uses the Hungarian algorithm on the confusion matrix to align predicted and true cluster IDs, returning overall accuracy.

### 3. `load_data(file_path)`

Loads a `.csv` file into a PyTorch float tensor.

### 4. `clear_output_folder(folder_path)`

Deletes all contents from a specified output folder to prepare for new visualizations.

### 5. `test_model(X_test, model, decoder, device)`

Runs the TabPFN model and Transformer decoder on a test set to predict centroids. Also times and logs the prediction step.

### 6. `load_true_cluster_ids(file_path)`

Loads true cluster IDs from a `.csv` file into a PyTorch long tensor.

### 7. `process_folder(test_folder, ground_truth_folder, exact_true_labels, model, decoder_state_dict, device, args)`

Main evaluation loop:

- Loads test datasets and centroids.
- Dynamically adjusts decoder output size.
- Predicts centroids and aligns dimensions.
- Computes:
  - Fixed-order loss between prediction and ground truth.
  - Clustering accuracy vs. true cluster labels.
- Generates and saves:
  - Centroid scatter plots.
  - Sorted centroid plots by L2 norm.

### 8. `main(args)`

- Loads model checkpoint (`new_data_model.pth`) including:
  - Trained TabPFN model state.
  - Decoder weights.
- Calls `process_folder()` to begin evaluation over all test datasets.

---

## Files and Directory Structure

- `new_data/test_no_header/.csv` Test tabular data
- `new_data/test_result/.txt` Ground truth centroids
- `new_data/test_w_exact_true_labels/*.csv` True cluster labels
- `new_data_model.pth` Trained model checkpoint
- `new_data_output/` Saved centroid plots

---

## Future Improvements

- Add batched evaluation for efficiency.
- Enable per-class error analysis and debugging.
- Improve centroid dimensionality matching (e.g., automatic projection).

---

## Usage

```bash
python test.py \
  --test_folder new_data/test_no_header \
  --ground_truth_folder new_data/test_result \
  --device cuda \
  --fit_mode low_memory \
  --static_seed 42 \
  --num_queries 2

```

---

## Authors

Part of the ERSP Research Project — UC San Diego  
Contributors: Nick Nguyen, Sophia Li, Myat Thiha, Kumiko Komori
