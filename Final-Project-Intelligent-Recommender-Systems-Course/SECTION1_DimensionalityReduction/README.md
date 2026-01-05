# Section 1: Dimensionality Reduction

## Overview

This section implements **Dimensionality Reduction** techniques for a recommendation system using **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**. The focus is on understanding latent factors in user-item rating data and evaluating robustness to missing data.

---

## Project Structure

```
SECTION1_DimensionalityReduction/
├── code/
│   ├── __init__.py
│   ├── svd_analysis.py     # Core SVD implementation and analysis
│   ├── pca_mean_filling.py # PCA with mean-filling strategies
│   ├── pca_mle.py          # PCA using Maximum Likelihood Estimation
│   └── utils.py            # Utility functions (metrics, data loading)
├── data/                   # Input datasets
├── plots/                  # Generated visualizations
├── results/                # Intermediate computation results
└── tables/                 # Output CSV tables
```

---

## Key Components

### 1. Manual SVD Implementation (`svd_analysis.py`)

- **Power Iteration**: Custom implementation for computing singular values
- **Manual SVD with Deflation**: Truncated SVD without relying on scipy
- **Rating Matrix Reconstruction**: Predict missing ratings using latent factors

### 2. Sensitivity Analysis

- Test robustness across varying missing data percentages (10%, 30%, 50%, 70%)
- Compare reconstruction error (MAE/RMSE) across different sparsity levels

### 3. Mean-Filling Strategies

- **Item Mean Filling**: Replace missing values with column averages
- **User Mean Filling**: Replace missing values with row averages
- **Comparison**: Evaluate which strategy yields better predictions

### 4. Latent Factor Interpretation

- Analyze the top-k latent factors extracted via SVD
- Interpret what each component represents (user/item clusters)

### 5. Cold Start Analysis

- **Fold-in Method**: Project new users into existing latent space
- Evaluate prediction accuracy for cold-start vs. warm-start users

---

## Utility Functions (`utils.py`)

| Function | Description |
|----------|-------------|
| `calculate_mae()` | Compute Mean Absolute Error |
| `calculate_rmse()` | Compute Root Mean Square Error |
| `load_data()` | Load pickle datasets |
| `get_subset()` | Filter users by minimum rating count |
| `create_rating_matrix()` | Pivot data to user-item matrix |
| `fill_with_item_mean()` | Fill NaN with item averages |
| `fill_with_user_mean()` | Fill NaN with user averages |
| `create_missing_data_mask()` | Generate random missing data mask |

---

## Output Files

### Tables (`tables/`)
- `Singular_values.csv` - Computed singular values
- `U_matrix_unique_users.csv` - User latent factor matrix
- `V_matrix_items.csv` - Item latent factor matrix
- `cold_start_results.csv` - Cold-start user predictions
- `warm_start_results.csv` - Warm-start user predictions
- `evaluation_results.csv` - MAE/RMSE evaluation metrics

---

## Usage

```python
from code.svd_analysis import manual_svd, run_sensitivity_analysis, run_cold_start_analysis
from code.utils import load_data, create_rating_matrix, fill_with_item_mean

# Load and prepare data
df = load_data('data/ratings.pkl')
rating_matrix = create_rating_matrix(df)
rating_filled = fill_with_item_mean(rating_matrix)

# Perform SVD with k=20 latent factors
U, S, Vt = manual_svd(rating_filled.values, k=20, verbose=True)

# Run sensitivity analysis
results = run_sensitivity_analysis(df, k=20, missing_pcts=[10, 30, 50, 70])

# Cold-start analysis
cold_results = run_cold_start_analysis(df, k=10, n_cold_users=50)
```

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scipy` (for comparison with library SVD)

---

## Authors

AIE425 Final Project - Group 22
