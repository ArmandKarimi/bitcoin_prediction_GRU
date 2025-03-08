import torch
import numpy as np
import pandas as pd


def inverse_transform(preds, test_data, window_size=30):
    """
    Convert normalized predictions back to real values using only past values.

    Args:
        preds (np.array or pd.Series): Normalized predictions from the model.
        test_data (pd.Series): Original (un-normalized) target values.
        window_size (int): Rolling window size (should match sequence length).

    Returns:
        np.array: Denormalized predictions.
    """
    denorm_preds = []

    for i, pred in enumerate(preds):
        # Ensure we have enough past data
        past_data = test_data.iloc[max(0, i-window_size):i]

        # If there's not enough data, use all available
        if len(past_data) == 0:
            past_data = test_data.iloc[:1]  # Avoid division by zero
        
        # Compute rolling mean and std using only past values (no lookahead)
        rolling_mean = past_data.mean()
        rolling_std = past_data.std() if past_data.std() > 0 else 1e-8  # Avoid division by zero

        # Apply inverse transformation
        real_value = (pred * rolling_std) + rolling_mean
        denorm_preds.append(real_value)

    return np.array(denorm_preds)




def evaluate(model, loader, device, original_data, window_size=24):
    """
    Evaluates the model on the given data loader and returns the predictions
    and ground truth values in the original scale.

    Parameters:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): The DataLoader for evaluation data.
        device (torch.device): The device to run the computations on.
        original_data (pd.Series): The original target data (e.g., df_test['Close']) for denormalization.
        window_size (int): The rolling window size used during normalization.

    Returns:
        tuple: (predictions, truths) as NumPy arrays.
    """
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for X, y in loader:
            # Move data to the correct device
            X, y = X.to(device), y.to(device)
            
            # Get model predictions and move to CPU as NumPy array
            preds = model(X).cpu().numpy()
            y = y.cpu().numpy()  # Convert ground truth to numpy
            
            # Inverse normalization for predictions and ground truth
            # preds = inverse_transform(preds, original_data, window_size)
            # y = inverse_transform(y, original_data, window_size)
            
            predictions.extend(preds.tolist())
            truths.extend(y.tolist())
    
    return np.array(predictions), np.array(truths)
