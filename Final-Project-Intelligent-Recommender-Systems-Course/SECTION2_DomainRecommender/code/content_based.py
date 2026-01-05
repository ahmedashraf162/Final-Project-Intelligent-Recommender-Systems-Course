
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import ast

class ContentBasedRecommender:
    """
    Content-Based Recommendation System using TF-IDF and Numerical Features.
    """
    
    def __init__(self):
        self.item_features = None
        self.recipe_id_to_idx = {}
        self.idx_to_recipe_id = {}
        self.tfidf_vectorizer = None
        self.scaler = None
        
    def fit(self, recipes_df):
        """
        Fit the model using recipe data.
        extracts text features (TF-IDF) and numerical/nutritional features.
        """
        print("\nTraining Content-Based Recommender...")
        
        # 1. TF-IDF on Combined Text
        print("  - Vectorizing text features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        # Ensure combined_text exists (it should from preprocessing)
        if 'combined_text' not in recipes_df.columns:
             # Basic fallback if not precomputed
             recipes_df['combined_text'] = recipes_df['name'].fillna('')  
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(recipes_df['combined_text'])
        tfidf_dense = tfidf_matrix.toarray()
        
        # 2. Numerical Features
        print("  - Processing numerical features...")
        
        def parse_nutrition(x):
            try:
                if pd.isna(x): return [0]*7
                return ast.literal_eval(x) if isinstance(x, str) else x
            except: return [0]*7

        nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
        # Check if nutrition is string or list
        if isinstance(recipes_df['nutrition'].iloc[0], str):
            nutrition_df = pd.DataFrame(recipes_df['nutrition'].apply(parse_nutrition).tolist(), columns=nutrition_cols)
        else:
             nutrition_df = pd.DataFrame(recipes_df['nutrition'].tolist(), columns=nutrition_cols)

        numerical_features = pd.DataFrame({
            'minutes': recipes_df['minutes'].fillna(recipes_df['minutes'].median()),
            'n_steps': recipes_df['n_steps'].fillna(recipes_df['n_steps'].median()),
            'n_ingredients': recipes_df['n_ingredients'].fillna(recipes_df['n_ingredients'].median())
        })
        
        additional_features = pd.concat([numerical_features, nutrition_df], axis=1)
        
        # Clip outliers
        additional_features['minutes'] = additional_features['minutes'].clip(upper=500)
        additional_features['calories'] = additional_features['calories'].clip(upper=5000)
        
        # Scale
        self.scaler = MinMaxScaler()
        additional_scaled = self.scaler.fit_transform(additional_features)
        
        # 3. Combine Features
        # Weight: 70% text, 30% additional
        text_weight = 0.7
        additional_weight = 0.3
        
        self.item_features = np.hstack([
            tfidf_dense * text_weight,
            additional_scaled * additional_weight
        ])
        
        # Mappings
        self.recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipes_df['id'])}
        self.idx_to_recipe_id = {idx: rid for rid, idx in self.recipe_id_to_idx.items()}
        
        print(f"Content-Based Model Trained. Feature matrix shape: {self.item_features.shape}")

    def build_user_profile(self, user_id, interactions_df):
        """Build user profile as weighted average of rated item features."""
        user_ratings = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        weighted_features = np.zeros(self.item_features.shape[1])
        total_weight = 0
        
        for _, row in user_ratings.iterrows():
            rid = row['recipe_id']
            rating = row['rating']
            
            if rid in self.recipe_id_to_idx:
                idx = self.recipe_id_to_idx[rid]
                weighted_features += self.item_features[idx] * rating
                total_weight += rating
                
        if total_weight > 0:
            return weighted_features / total_weight
        return None

    def recommend(self, user_id, interactions_df, n=10):
        """Generate top-N recommendations for a user."""
        user_profile = self.build_user_profile(user_id, interactions_df)
        
        if user_profile is None:
            # Fallback or empty (caller should handle cold start if strict)
            return pd.DataFrame()
            
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Exclude rated items
        rated_items = set(interactions_df[interactions_df['user_id'] == user_id]['recipe_id'].values)
        
        # Sort indices
        sorted_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            rid = self.idx_to_recipe_id[idx]
            if rid not in rated_items:
                recommendations.append({
                    'recipe_id': rid,
                    'score': similarities[idx],
                    'method': 'Content-Based'
                })
                if len(recommendations) >= n:
                    break
                    
        return pd.DataFrame(recommendations)
