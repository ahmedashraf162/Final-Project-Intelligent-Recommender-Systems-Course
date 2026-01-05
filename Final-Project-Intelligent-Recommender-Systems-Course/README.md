# AIE425 Final Project - Recommendation Systems

## Group 22

A comprehensive recommendation system project implementing dimensionality reduction techniques and domain-specific recipe recommendations.

---

## Project Structure

```
AIE425_FinalProject_Group-22-/
├── SECTION1_DimensionalityReduction/  # SVD/PCA Analysis
│   ├── code/                          # Implementation modules
│   ├── data/                          # Input datasets
│   ├── plots/                         # Generated visualizations
│   ├── results/                       # Computation results
│   └── tables/                        # Output CSV tables
│
├── SECTION2_DomainRecommender/        # Recipe Recommendation System
│   ├── code/                          # Recommender implementations
│   ├── data/                          # Recipe & interaction data
│   └── results/                       # Recommendation outputs
│
├── Statistical_Analysis.ipynb         # Statistical analysis notebook
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## Section 1: Dimensionality Reduction

Implements **SVD** and **PCA** for understanding latent factors in user-item rating data.

### Key Features
- Manual SVD implementation using power iteration.
- Sensitivity analysis across varying missing data percentages.
- Comparison of mean-filling strategies (item-mean vs. user-mean).
- Latent factor interpretation and visualization.
- Cold-start user analysis with fold-in method.

### Main Modules
| File | Description |
|------|-------------|
| `svd_analysis.py` | Core SVD implementation and analysis functions |
| `pca_mean_filling.py` | PCA with mean-filling strategies |
| `pca_mle.py` | PCA using Maximum Likelihood Estimation |
| `utils.py` | Metrics (MAE/RMSE), data loading, preprocessing |

---

## Section 2: Domain Recommender

Implements a **Recipe Recommendation System** with multiple strategies.

### Recommendation Approaches

| Approach | Description |
|----------|-------------|
| **Content-Based** | TF-IDF on recipe text + numerical/nutritional features |
| **Collaborative Filtering** | Item-based CF + SVD matrix factorization |
| **Hybrid (Switching)** | Uses CB for cold-start, CF for active users |

### Main Modules
| File | Description |
|------|-------------|
| `main.py` | Streamlit web application |
| `data_preprocessing.py` | Data loading, cleaning, and scaling |
| `content_based.py` | Content-Based Recommender class |
| `collaborative.py` | Collaborative Filtering class (CF + SVD) |
| `hybrid.py` | Hybrid Switching Recommender |

### Run the Web App

```bash
cd SECTION2_DomainRecommender/code
streamlit run main.py
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AIE425_FinalProject_Group-22-
   ```

2. **Create virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Section 1 - SVD Analysis
```python
from SECTION1_DimensionalityReduction.code.svd_analysis import manual_svd, run_cold_start_analysis
from SECTION1_DimensionalityReduction.code.utils import load_data, create_rating_matrix

# Load data and perform SVD
df = load_data('SECTION1_DimensionalityReduction/data/ratings.pkl')
rating_matrix = create_rating_matrix(df)
U, S, Vt = manual_svd(rating_matrix.fillna(0).values, k=20)
```

### Section 2 - Recipe Recommendations
```python
from SECTION2_DomainRecommender.code.content_based import ContentBasedRecommender
from SECTION2_DomainRecommender.code.data_preprocessing import load_data, preprocess_recipes

# Load and prepare data
interactions, recipes = load_data('SECTION2_DomainRecommender/data')
recipes = preprocess_recipes(recipes)

# Train and get recommendations
model = ContentBasedRecommender()
model.fit(recipes)
recommendations = model.recommend(user_id=12345, interactions_df=interactions, n=10)
```

---

## Dependencies

See `requirements.txt` for full list. Main packages:

- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`
- **Web App**: `streamlit`

---

## Authors

**Group 22** - AIE425 Final Project
