import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# --- Metrics ---

def calculate_mae(original, approximation, mask=None):
    """
    Calculate Mean Absolute Error (MAE).
    """
    if hasattr(original, 'values'):
        original = original.values
    if hasattr(approximation, 'values'):
        approximation = approximation.values
        
    if mask is not None:
        if hasattr(mask, 'values'):
            mask = mask.values
        if mask.any():
            return np.mean(np.abs(original[mask] - approximation[mask]))
        else:
            return 0.0
    else:
        return np.mean(np.abs(original - approximation))

def calculate_rmse(original, approximation, mask=None):
    """
    Calculate Root Mean Square Error (RMSE).
    """
    if hasattr(original, 'values'):
        original = original.values
    if hasattr(approximation, 'values'):
        approximation = approximation.values
        
    if mask is not None:
        if hasattr(mask, 'values'):
            mask = mask.values
        if mask.any():
            return np.sqrt(np.mean((original[mask] - approximation[mask])**2))
        else:
            return 0.0
    else:
        return np.sqrt(np.mean((original - approximation)**2))

# --- Data Loading ---

def load_data(file_path):
    """Load pickle data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_subset(df, min_user_ratings=20, n_samples=None, random_state=42):
    """Filter dataset for users with minimum ratings."""
    user_counts = df['user-id'].value_counts()
    valid_users = user_counts[user_counts > min_user_ratings].index
    
    df_subset = df[df['user-id'].isin(valid_users)]
    if n_samples and len(df_subset) > n_samples:
        df_subset = df_subset.sample(n=n_samples, random_state=random_state)
    return df_subset

# --- Preprocessing & Filling ---

def create_rating_matrix(df):
    """Pivot to user-item matrix."""
    return df.pivot(index='user-id', columns='item-id', values='rating')

def fill_with_item_mean(rating_matrix, item_means=None):
    """Fill NaNs with column means."""
    filled = rating_matrix.copy()
    if item_means is not None:
        for col in filled.columns:
            if col in item_means:
                filled[col] = filled[col].fillna(item_means[col])
    else:
        filled = filled.apply(lambda col: col.fillna(col.mean()), axis=0)
    return filled.fillna(0)

def fill_with_user_mean(rating_matrix):
    """Fill NaNs with row means."""
    filled = rating_matrix.copy()
    filled = filled.apply(lambda row: row.fillna(row.mean()), axis=1)
    return filled.fillna(0)

def create_missing_data_mask(rating_matrix, missing_pct, random_state=None):
    """Create mask for testing."""
    if random_state:
        np.random.seed(random_state)
    mask = np.random.rand(*rating_matrix.shape) < (missing_pct / 100.0)
    mask = mask & (~rating_matrix.isna())
    masked_matrix = rating_matrix.copy()
    masked_matrix[mask] = np.nan
    return masked_matrix, mask
