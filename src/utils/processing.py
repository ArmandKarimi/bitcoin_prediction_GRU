import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import logging
import pandas as pd
import numpy as np
from config import RATIOS, SEQ_LENGTH, PRED_LENGTH, BATCH_SIZE 

#____SPLIT data chornolgically_______
def chronological_split(data, ratios=(0.7, 0.15, 0.15), buffer=7):
    # Split ratios
    
    """
    Splits the DataFrame into training, validation, and test sets in chronological order with a buffer.
    """
    n = len(data)
    train_end = int(n * ratios[0]) - buffer
    val_end = train_end + int(n * ratios[1]) - buffer

    df_train = data.iloc[:train_end]
    df_val = data.iloc[train_end+buffer:val_end]
    df_test = data.iloc[val_end+buffer:]

    return df_train, df_val, df_test



#------since data is non-stationary-------
def moving_avg_normalization(data, window_size=24):
    normalized_data = data.copy()
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # Ensure future data is not included in the moving window
            shifted_col = data[column].shift(1)  # Shift by 1 to exclude current value
            rolling_mean = shifted_col.rolling(window=window_size, min_periods=1).mean()
            rolling_std = shifted_col.rolling(window=window_size, min_periods=1).std()

            # Normalize data
            normalized_data[column] = (data[column] - rolling_mean) / (rolling_std + 1e-8)
    normalized_data = normalized_data.dropna()
    return normalized_data


#------- create sequences -------
def create_sequences(X, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH):
    """
    Creates sequences and corresponding targets from a DataFrame.
    
    Parameters:
        X (pd.DataFrame): The input data containing features. It must include a column 'Close' 
                          for the target variable.
        seq_length (int): The length of the input sequence.
        pred_length (int): The number of future time steps to predict.
        
    Returns:
        tuple: (X_seq, y_seq) where:
            - X_seq is a torch.Tensor of shape (num_sequences, seq_length, num_features)
            - y_seq is a torch.Tensor of shape (num_sequences, pred_length)
    """
    sequences = []
    targets = []
    
    # Ensure there are enough rows to create at least one sequence
    for i in range(len(X) - seq_length - pred_length + 1):
        # Get the sequence of features; this takes all columns for rows i to i+seq_length
        sequences.append(X.iloc[i:i+seq_length].values)
        # Get the target sequence from the 'Close' column only
        targets.append(X['Close'].iloc[i+seq_length:i+seq_length+pred_length].values)
    
    X_seq = torch.tensor(sequences, dtype=torch.float32)
    y_seq = torch.tensor(targets, dtype=torch.float32)
    return X_seq, y_seq

#----create data loaders------
def data_loader(X_seq, y_seq, batch_size=BATCH_SIZE):
    """
    Creates a DataLoader for the given sequences and targets.
    Parameters:
        X_seq (torch.Tensor): Tensor containing input sequences.
        y_seq (torch.Tensor): Tensor containing corresponding target sequences.
        batch_size (int): Batch size for the DataLoader.
        
    Returns:
        DataLoader: A DataLoader wrapping a TensorDataset of (X_seq, y_seq).
    """
    dataset = TensorDataset(X_seq, y_seq)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)






if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    from fetch_data import load_data
    from config import NAME
      # --- Data Fetching ---
    logger.info("Fetching data...")
    data = load_data(name = NAME)
    logger.info(f"✅ Data fetched. Shape: {data.shape}")
    
    # --- Data Processing ---
    df_train, df_val, df_test = chronological_split(data)
    logger.info("✅ Data split into train, validation, and test sets.")
    
    X_train = moving_avg_normalization(df_train)
    X_val = moving_avg_normalization(df_val)
    X_test = moving_avg_normalization(df_test)
    logger.info("✅ Data normalized.")
    
    X_train_seq, y_train_seq = create_sequences(X_train, SEQ_LENGTH, PRED_LENGTH)
    X_val_seq, y_val_seq     = create_sequences(X_val, SEQ_LENGTH, PRED_LENGTH)
    X_test_seq, y_test_seq   = create_sequences(X_test, SEQ_LENGTH, PRED_LENGTH)
    logger.info(f"✅ Sequences created: Training: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    # Create DataLoaders
    train_loader = data_loader(X_train_seq, y_train_seq, BATCH_SIZE)
    val_loader   = data_loader(X_val_seq, y_val_seq, BATCH_SIZE)
    test_loader  = data_loader(X_test_seq, y_test_seq, BATCH_SIZE)

 
    print("X_train_seq.shape = ", X_train_seq.shape)
    print("y_train_seq.shape = ", y_train_seq.shape)

    