
import pandas as pd
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender

class HybridRecommender:
    """
    Switching Hybrid Recommender System.
    Switches between Content-Based (Cold Start) and Collaborative Filtering (Active Users).
    """
    
    def __init__(self, content_model: ContentBasedRecommender, cf_model: CollaborativeRecommender):
        self.content_model = content_model
        self.cf_model = cf_model
        self.rating_threshold = 10
        
    def recommend(self, user_id, interactions_df, n=10):
        """
        Generate hybrid recommendations based on user activity.
        """
        # Determine user activity level
        user_history = interactions_df[interactions_df['user_id'] == user_id]
        rating_count = len(user_history)
        
        # Switching Logic
        if rating_count < self.rating_threshold:
            print(f"User {user_id} is Cold-Start ({rating_count} ratings). Using Content-Based.")
            recommendations = self.content_model.recommend(user_id, interactions_df, n)
            if not recommendations.empty:
                recommendations['method'] = 'Content-Based (Cold-Start)'
            return recommendations
        else:
            print(f"User {user_id} is Active ({rating_count} ratings). Using Collaborative Filtering.")
            # Try CF
            recommendations = self.cf_model.recommend(user_id, n, method='svd')
            
            # Fallback if CF fails (e.g. user filtered out during matrix construction)
            if recommendations.empty:
                print(f"User {user_id} not found in CF model. Falling back to Content-Based.")
                recommendations = self.content_model.recommend(user_id, interactions_df, n)
                if not recommendations.empty:
                    recommendations['method'] = 'Content-Based (Fallback)'
            
            return recommendations
