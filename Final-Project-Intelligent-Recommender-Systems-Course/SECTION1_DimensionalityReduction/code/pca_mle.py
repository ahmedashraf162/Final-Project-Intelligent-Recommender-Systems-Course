import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .utils import calculate_mae, calculate_rmse, fill_with_item_mean, create_rating_matrix

def perform_pca_mle_analysis(df):
    """
    Execute the PCA MLE analysis from Part 2.
    """
    print("--- Starting PCA MLE Analysis ---")
    
    # 1. Create Matrix
    R = create_rating_matrix(df)
    
    # 2. Fill Missing
    R_filled = fill_with_item_mean(R)
    
    # 3. PCA with MLE
    print("Fitting PCA with n_components='mle'...")
    pca = PCA(n_components='mle')
    P = pca.fit_transform(R_filled)
    
    print(f"MLE selected {pca.n_components_} components.")
    
    # 4. Reconstruct
    R_approx = pca.inverse_transform(P)
    
    # 5. Evaluate
    mae = calculate_mae(R_filled.values, R_approx)
    rmse = calculate_rmse(R_filled.values, R_approx)
    
    print(f"MLE Reconstruction Error - MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    return pca, P, R_approx
