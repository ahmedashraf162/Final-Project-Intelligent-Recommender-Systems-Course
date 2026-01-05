import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import (calculate_mae, calculate_rmse, fill_with_item_mean, 
                   fill_with_user_mean, create_rating_matrix, create_missing_data_mask,
                   get_subset)

# --- Core SVD Logic (Manual) ---

def power_iteration(A, num_iterations=100, tol=1e-10):
    m, n = A.shape
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        u = A @ v
        sigma = np.linalg.norm(u)
        if sigma < tol:
            return 0, np.zeros(m), np.zeros(n)
        u = u / sigma
        v_new = A.T @ u
        v_new = v_new / np.linalg.norm(v_new)
        if np.abs(np.abs(np.dot(v, v_new)) - 1) < tol:
            break
        v = v_new
    
    return sigma, u, v

def manual_svd(R, k, verbose=False):
    """
    Perform truncated SVD manually using deflation.
    """
    m, n = R.shape
    U = np.zeros((m, k))
    S = np.zeros(k)
    V = np.zeros((n, k))
    
    R_residual = R.copy()
    
    for i in range(k):
        sigma, u, v = power_iteration(R_residual)
        U[:, i] = u
        S[i] = sigma
        V[:, i] = v
        R_residual = R_residual - sigma * np.outer(u, v)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Computed {i + 1}/{k} singular values...")
    
    return U, S, V

# --- Sensitivity Analysis ---

def run_sensitivity_analysis(df, k=50, missing_pcts=[10, 30, 50, 70]):
    print("--- Starting Sensitivity Analysis ---")
    R = create_rating_matrix(df)
    
    results = {'missing_pct': [], 'recon_mae': [], 'pred_mae': []}
    
    for pct in missing_pcts:
        print(f"  Processing missing %: {pct}")
        masked_df, mask = create_missing_data_mask(R, pct, random_state=42)
        
        # Using Item Mean Filling for Sensitivity Base Case
        filled_df = fill_with_item_mean(masked_df)
        R_filled = filled_df.values
        
        U, S, V = manual_svd(R_filled, k)
        R_approx = U @ np.diag(S) @ V.T
        
        # Errors
        # Prediction Error: Compare Approx vs Original (Ground Truth) at MASKED locations
        # Note: Original R might have NaNs. We assume df passed is ground truth subset.
        R_original = R.fillna(0).values 
        
        recon_mae = calculate_mae(R_filled, R_approx)
        pred_mae = calculate_mae(R_original, R_approx, mask)
        
        results['missing_pct'].append(pct)
        results['recon_mae'].append(recon_mae)
        results['pred_mae'].append(pred_mae)
        
    return results

def compare_mean_filling_strategies(df, k=50, missing_pct=30):
    print("--- Comparing Mean Filling Strategies ---")
    R = create_rating_matrix(df)
    masked_df, mask = create_missing_data_mask(R, missing_pct, random_state=42)
    R_original = R.fillna(0).values
    
    # 1. Item Mean
    filled_item = fill_with_item_mean(masked_df).values
    U, S, V = manual_svd(filled_item, k)
    approx_item = U @ np.diag(S) @ V.T
    mae_item = calculate_mae(R_original, approx_item, mask)
    
    # 2. User Mean
    filled_user = fill_with_user_mean(masked_df).values
    U, S, V = manual_svd(filled_user, k)
    approx_user = U @ np.diag(S) @ V.T
    mae_user = calculate_mae(R_original, approx_user, mask)
    
    print(f"Prediction MAE (Item Mean): {mae_item:.6f}")
    print(f"Prediction MAE (User Mean): {mae_user:.6f}")
    
    return {'Item Mean': mae_item, 'User Mean': mae_user}

# --- Latent Factor Interpretation ---

def interpret_latent_factors(U, S, Vt, df, k=3):
    """
    Simulate the component interpretation from Part 3.
    """
    print("--- Interpreting Latent Factors ---")
    
    # We need the alignment of indices/columns from the matrix
    # Re-create R to get index maps (efficient if just for headers)
    R_ref = create_rating_matrix(df)
    item_ids = R_ref.columns
    user_ids = R_ref.index
    
    # Transpose Vt to get V (Items x Factors)
    V = Vt.T 
    
    factor_analysis = []
    
    # Analyze Top Factors
    for i in range(min(k, U.shape[1])):
        factor_num = i + 1
        
        # Top Items (largest magnitude in V column)
        # Note: V is (n_items, n_components)
        top_items_idx = np.argsort(np.abs(V[:, i]))[::-1][:10]
        top_items = item_ids[top_items_idx]
        top_items_values = V[top_items_idx, i]
        
        # Top Users (largest magnitude in U column)
        top_users_idx = np.argsort(np.abs(U[:, i]))[::-1][:10]
        top_users = user_ids[top_users_idx]
        top_users_values = U[top_users_idx, i]
        
        # Stats
        # Use R_ref instead of the filled matrix for stats calculation
        top_items_avg = R_ref[top_items].mean()
        top_items_count = R_ref[top_items].count()
        
        factor_analysis.append({
            'factor': factor_num,
            'top_items': list(zip(top_items, top_items_values)),
            'top_users': list(zip(top_users, top_users_values)),
            'stats': (top_items_avg, top_items_count)
        })
        
    # Print Report
    for f in factor_analysis:
        print(f"\nFactor {f['factor']}:")
        print("  Top 10 Items (ID: Weight):")
        for item, val in f['top_items']:
            print(f"    {item}: {val:.4f}")
        print("  Top 10 Users (ID: Weight):")
        for user, val in f['top_users']:
            print(f"    {user}: {val:.4f}")
            
    return factor_analysis

# --- Cold Start Analysis ---

def predict_user_ratings(r_u_row, V, S):
    """
    Fold-in method: Project user row onto existing V space.
    u_new = r_u @ V @ S_inv
    pred = u_new @ S @ Vt
    """
    # S_inv: Inverse of singular values diagonal
    # Handle zeros in S to avoid div/0
    S_inv_diag = np.zeros_like(S)
    mask = S > 1e-12
    S_inv_diag[mask] = 1.0 / S[mask]
    S_inv = np.diag(S_inv_diag)
    
    # Project
    # r_u shape: (1, n_items), V: (n_items, k), S_inv: (k, k)
    u_factors = r_u_row @ V @ S_inv
    
    # Reconstruct
    # u_factors: (1, k), diag(S): (k, k), Vt: (k, n_items)
    pred_ratings = u_factors @ np.diag(S) @ V.T
    
    return pred_ratings

def run_cold_start_analysis(df, k=10, n_cold_users=50, hide_pct=0.8):
    print(f"--- Starting Cold Start Analysis (k={k}) ---")
    import math # Verify import inside function to avoid top-level clutter if preferred
    
    # 1. Prepare Full Matrix and Model
    R_df = create_rating_matrix(df)
    
    # Fill standard missing values for 'training' the base model
    # (In a real scenario, we might ignore cold users in training, but following notebook logic)
    R_filled = fill_with_item_mean(R_df)
    R_matrix = R_filled.values
    
    # Full SVD (Manual)
    print("  Training SVD model on full data...")
    U, S, Vt = manual_svd(R_matrix, k)
    V = Vt.T # (Items, k)
    
    # 2. Select Cold Users
    user_counts = R_df.count(axis=1)
    eligible_users = user_counts[user_counts > 20].index
    
    if len(eligible_users) == 0:
        return "No eligible users for cold start."
        
    actual_n = min(n_cold_users, len(eligible_users))
    cold_users = np.random.choice(eligible_users, size=actual_n, replace=False)
    print(f"  Selected {actual_n} users for cold-start simulation.")
    
    mae_list = []
    rmse_list = []
    
    # 3. Simulate and Predict
    for uid in cold_users:
        u_idx = R_df.index.get_loc(uid)
        
        # Get ground truth row
        original_row = R_df.iloc[u_idx] # Has NaNs
        rated_items_indices = np.where(~original_row.isna())[0]
        
        if len(rated_items_indices) < 5: continue # Skip if too few ratings despite check
        
        # Hide Items
        n_hide = int(len(rated_items_indices) * hide_pct)
        if n_hide == 0: n_hide = 1
        
        hidden_indices = np.random.choice(rated_items_indices, size=n_hide, replace=False)
        visible_indices = np.setdiff1d(rated_items_indices, hidden_indices)
        
        # Construct r_input: same as Filled row, but with HIDDEN items set to 0 (or mean?)
        # Notebook sets hidden to NaN, then fillna(0). So Hidden -> 0. Visible -> Real Value.
        # This implies we treat hidden as unrated.
        
        r_input = R_filled.iloc[u_idx].copy().values
        # But wait, R_filled has item means for unrated. 
        # We want to emulate "We don't know these hidden ratings".
        # So for hidden indices, do we put 0 or Item Mean?
        # Notebook: R_cold.loc[u_id, hidden] = np.nan -> fillna(0). So 0.
        
        r_input[hidden_indices] = 0 # effectively hiding them and treating as zero-rating
        # Note: If model was trained on Mean-Filled, projecting a Zero-filled vector might be mismatch 
        # but following notebook extraction.
        
        # Predict
        pred_row = predict_user_ratings(r_input, V, S)
        
        # Evaluate on HIDDEN items
        true_vals = original_row.iloc[hidden_indices].values
        pred_vals = pred_row[hidden_indices]
        
        # Metrics
        mae = np.mean(np.abs(true_vals - pred_vals))
        rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        
    avg_mae = np.mean(mae_list) if mae_list else 0
    avg_rmse = np.mean(rmse_list) if rmse_list else 0
    
    print(f"Cold Start Results - MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")
    
    return {'mae': avg_mae, 'rmse': avg_rmse, 'users_tested': actual_n}
