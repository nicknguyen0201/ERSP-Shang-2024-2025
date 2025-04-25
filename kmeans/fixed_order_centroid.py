import torch
import torch.nn.functional as F
import re


def parse_centroids(file_path, device):
    """
    Parses a text file containing centroids enclosed in square brackets and converts them into a PyTorch tensor.
    """
    with open(file_path, "r") as file:
        content = file.read()

    centroid_strings = re.findall(r"\[([^\]]+)\]", content)
    centroids = []

    for s in centroid_strings:
        float_values = [float(num_str) for num_str in s.split()]
        tensor_value = torch.tensor(float_values, dtype=torch.float32).to(device)
        centroids.append(tensor_value)

    if not centroids:
        return torch.empty(0, device=device)

    return torch.stack(centroids).to(device)


def fixed_order_centroid_loss(predicted, target):
    """
    Computes MSE loss after sorting both predicted and target centroids by their distance from the origin.
    """
    pred_norms = torch.norm(predicted, dim=1)
    targ_norms = torch.norm(target, dim=1)

    pred_sorted = predicted[torch.argsort(pred_norms)]
    target_sorted = target[torch.argsort(targ_norms)]

    base_loss = F.mse_loss(pred_sorted, target_sorted)

    penalty_weight = 1e-9
    pairwise_dists = pred_sorted.unsqueeze(1) - pred_sorted.unsqueeze(0)
    norms = torch.norm(pairwise_dists, dim=2)
    norms_upper_half = torch.triu(norms, diagonal=1)
    penalty = torch.sum(1 / (norms_upper_half + 1e-6))

    print("Pred norm order:", pred_norms.tolist())
    print("GT norm order:", targ_norms.tolist())

    loss = base_loss + penalty_weight * penalty
    print(f"[Fixed Order] Base Loss: {base_loss.item()}, Penalty: {penalty_weight * penalty.item()}")

    return loss


def visualize_centroids(predictions, ground_truth, X, save_path):
    """
    Sorts predictions and ground truth centroids by norm and saves a plot.
    """
    from plot_centroids import plot_centroids

    pred_sorted = predictions[torch.argsort(torch.norm(predictions, dim=1))].detach().cpu().numpy()
    gt_sorted = ground_truth[torch.argsort(torch.norm(ground_truth, dim=1))].detach().cpu().numpy()
    plot_centroids(X, pred_sorted, gt_sorted, save_path)
