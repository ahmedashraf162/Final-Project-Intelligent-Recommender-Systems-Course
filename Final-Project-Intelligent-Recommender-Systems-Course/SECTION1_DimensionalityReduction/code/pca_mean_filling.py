import numpy as np
import pandas as pd
from .utils import calculate_mae, calculate_rmse, fill_with_item_mean, create_rating_matrix

def perform_pca_mean_filling_analysis(df, n_components=None):
    """
    Execute the PCA Mean Filling analysis from Part 1.
    """
    print("--- Starting PCA Mean Filling Analysis ---")
    
    # 1. Create Matrix
    R = create_rating_matrix(df)
    
    # 2. Fill Missing (Item Mean)
    R_filled = fill_with_item_mean(R)
    R_matrix = R_filled.values
    
    # 3. PCA (via SVD)
    # Part 1 uses np.linalg.svd
    U, s, Vt = np.linalg.svd(R_matrix, full_matrices=False)
    
    # If n_components specified, reconstruction would use truncated SVD
    # The notebook saves U, s, Vt. 
    
    print(f"SVD computed. Shapes: U={U.shape}, s={s.shape}, Vt={Vt.shape}")
    
    # Reconstruction logic (Standard PCA reconstruction using all components effectively gives exact match)
    # If we want to verify reconstruction error with *full* components (checking numerical stability):
    Sigma = np.diag(s)
    R_reconstructed = U @ Sigma @ Vt
    
    mae = calculate_mae(R_matrix, R_reconstructed)
    rmse = calculate_rmse(R_matrix, R_reconstructed)
    
    print(f"Full Reconstruction Error - MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    return U, s, Vt, R_reconstructed
