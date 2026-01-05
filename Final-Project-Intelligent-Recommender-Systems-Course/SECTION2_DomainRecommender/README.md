# Section 2: Domain Recommender System

## Overview

This section implements a **Recipe Recommendation System** using multiple recommendation strategies:

- **Content-Based Filtering** (TF-IDF + Numerical Features)
- **Collaborative Filtering** (Item-Based CF + SVD Matrix Factorization)
- **Hybrid Switching System** (Combines CB and CF based on user activity)

The system includes an interactive **Streamlit web application** for real-time recipe recommendations.

---

## Project Structure

```
SECTION2_DomainRecommender/
├── code/
│   ├── main.py               # Streamlit app entry point
│   ├── data_preprocessing.py # Data loading and cleaning
│   ├── content_based.py      # Content-Based Recommender class
│   ├── collaborative.py      # Collaborative Filtering class
│   └── hybrid.py             # Hybrid Switching Recommender
├── data/
│   ├── RAW_interactions.csv  # User-recipe interactions
│   └── RAW_recipes.csv       # Recipe metadata
└── results/                  # Output recommendations
```

---

## Key Components

### 1. Data Preprocessing (`data_preprocessing.py`)

- **Load datasets**: Raw interactions and recipes
- **Clean interactions**: Handle missing values, remove duplicates
- **Scale ratings**: Normalize to 1-5 range
- **Process recipes**: Parse ingredients/tags, create combined text for TF-IDF

### 2. Content-Based Recommender (`content_based.py`)

**Features extracted:**
| Feature Type | Details |
|--------------|---------|
| Text (TF-IDF) | Recipe name, description, ingredients, tags |
| Numerical | Cooking time, steps count, ingredient count |
| Nutritional | Calories, fat, sugar, sodium, protein, carbs |

**Algorithm:**
1. Vectorize text using TF-IDF (500 features, n-grams 1-2)
2. Scale numerical features with MinMaxScaler
3. Combine: 70% text weight + 30% numerical weight
4. Build user profile as weighted average of rated items
5. Recommend using cosine similarity

### 3. Collaborative Filtering (`collaborative.py`)

**Item-Based CF:**
- Compute item-item cosine similarity matrix
- Predict ratings based on similar item ratings

**SVD Matrix Factorization:**
- Mean-fill missing ratings
- Perform truncated SVD (default k=20)
- Reconstruct predictions and clip to valid range [1, 5]

### 4. Hybrid Recommender (`hybrid.py`)

**Switching Strategy:**
- **Cold-start users** (< 10 ratings) → Content-Based
- **Active users** (≥ 10 ratings) → Collaborative Filtering (SVD)
- **Fallback**: If CF fails, revert to Content-Based

---

## Streamlit Application (`main.py`)

### Features

1. **Text-Based Input**: Enter preferences like "quick healthy chicken dinner"
2. **Rating-Based Input**: Rate recipes to build a preference profile
3. **Recommendations**: Get personalized top-N recipe suggestions

### Run the App

```bash
cd SECTION2_DomainRecommender/code
streamlit run main.py
```

---

## Usage Example

```python
from data_preprocessing import load_data, clean_interactions, preprocess_recipes
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

# Load and preprocess data
interactions, recipes = load_data('../data')
interactions = clean_interactions(interactions)
recipes = preprocess_recipes(recipes)

# Train models
cb_model = ContentBasedRecommender()
cb_model.fit(recipes)

cf_model = CollaborativeRecommender()
cf_model.prepare_data(interactions)
cf_model.train_svd(k=20)

# Create hybrid system
hybrid = HybridRecommender(cb_model, cf_model)

# Get recommendations for a user
recommendations = hybrid.recommend(user_id=12345, interactions_df=interactions, n=10)
print(recommendations)
```

---

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `streamlit`

---

## Authors

AIE425 Final Project - Group 22
