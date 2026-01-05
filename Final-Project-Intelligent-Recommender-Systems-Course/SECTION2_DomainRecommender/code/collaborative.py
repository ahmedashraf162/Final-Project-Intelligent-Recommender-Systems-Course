
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

class CollaborativeRecommender:
    """
    Collaborative Filtering using Item-Based CF and Matrix Factorization (SVD).
    """
    
    def __init__(self):
        self.rating_matrix = None
        self.item_similarity_df = None
        self.cf_predictions = None
        self.global_mean = 0
        
    def prepare_data(self, interactions_df, min_user_ratings=10, min_item_ratings=5):
        """
        Filter data and create rating matrix.
        """
        print("\nPreparing Collaborative Filtering Data...")
        
        # Filter active users and popular items
        user_counts = interactions_df['user_id'].value_counts()
        item_counts = interactions_df['recipe_id'].value_counts()
        
        active_users = user_counts[user_counts >= min_user_ratings].index
        popular_items = item_counts[item_counts >= min_item_ratings].index
        
        filtered_interactions = interactions_df[
            (interactions_df['user_id'].isin(active_users)) & 
            (interactions_df['recipe_id'].isin(popular_items))
        ]
        
        # Create Pivot Table
        self.rating_matrix = filtered_interactions.pivot_table(
            index='user_id', 
            columns='recipe_id', 
            values='rating',
            aggfunc='mean'
        )
        
        print(f"Rating Matrix Shape: {self.rating_matrix.shape}")
        self.global_mean = self.rating_matrix.stack().mean()

    def train_item_cf(self):
        """
        Compute Item-Item Similarity matrix.
        """
        print("Training Item-Based CF...")
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Call prepare_data first.")
            
        # Fill NaN with 0 for cosine similarity
        rating_matrix_filled = self.rating_matrix.fillna(0)
        
        similarity_matrix = cosine_similarity(rating_matrix_filled.T)
        self.item_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.rating_matrix.columns,
            columns=self.rating_matrix.columns
        )
        print("Item Similarity Matrix Computed.")

    def train_svd(self, k=20):
        """
        Perform truncated SVD Matrix Factorization.
        """
        print(f"Training SVD (k={k})...")
        if self.rating_matrix is None:
            raise ValueError("Rating matrix not built. Call prepare_data first.")
            
        # Mean filling
        item_means = self.rating_matrix.mean(axis=0)
        rating_filled = self.rating_matrix.apply(lambda x: x.fillna(item_means), axis=1)
        rating_filled = rating_filled.fillna(self.global_mean)
        
        # Center the matrix
        R = rating_filled.values
        user_ratings_mean = np.mean(R, axis=1)
        R_centered = R - user_ratings_mean.reshape(-1, 1)
        
        # SVD
        U, sigma, Vt = svds(R_centered, k=k)
        sigma = np.diag(sigma)
        
        # Reconstruct
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        
        # Clip to valid range
        all_user_predicted_ratings = np.clip(all_user_predicted_ratings, 1, 5)
        
        self.cf_predictions = pd.DataFrame(
            all_user_predicted_ratings,
            index=self.rating_matrix.index,
            columns=self.rating_matrix.columns
        )
        print("SVD Training Complete.")

    def recommend(self, user_id, n=10, method='svd'):
        """
        Generate recommendations using specified method.
        """
        if method == 'svd':
            return self._recommend_svd(user_id, n)
        else:
            print(f"Method {method} not implemented in main interface, using SVD.")
            return self._recommend_svd(user_id, n)

    def _recommend_svd(self, user_id, n):
        if self.cf_predictions is None or user_id not in self.cf_predictions.index:
            return pd.DataFrame() # User not in latent factor model
            
        user_preds = self.cf_predictions.loc[user_id]
        
        # Exclude rated items (using the original rating matrix to identify rated items)
        # Note: rating_matrix has NaN for unrated
        already_rated = self.rating_matrix.loc[user_id].dropna().index
        
        # Filter unrated
        unrated_preds = user_preds.drop(labels=already_rated, errors='ignore')
        
        # Top N
        top_n = unrated_preds.nlargest(n)
        
        return pd.DataFrame({
            'recipe_id': top_n.index,
            'score': top_n.values,
            'method': 'Collaborative Filtering (SVD)'
        })
