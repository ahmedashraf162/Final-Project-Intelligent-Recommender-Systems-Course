
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data as load_raw_data, clean_interactions, scale_ratings, preprocess_recipes
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

# Page Config
st.set_page_config(
    page_title="Recipe Recommender",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .recipe-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .score-badge {
        background-color: #FFE66D;
        color: #333;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prep_data():
    """Load and preprocess data using the modular data_preprocessing script."""
    # Determine data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # Load raw data
    interactions, recipes = load_raw_data(data_dir)
    
    # Preprocess
    # Note: Using st.spinner in the main execution flow, not inside cached func if possible,
    # but this is cached so it runs once.
    interactions = clean_interactions(interactions)
    interactions = scale_ratings(interactions)
    recipes = preprocess_recipes(recipes)
    
    return recipes, interactions

@st.cache_resource
def init_models(_recipes, _interactions):
    """Initialize and train models."""
    
    # Content Based
    cb_model = ContentBasedRecommender()
    cb_model.fit(_recipes)
    
    # Collaborative Filtering
    cf_model = CollaborativeRecommender()
    cf_model.prepare_data(_interactions)
    # Train SVD
    cf_model.train_svd(k=20)
    
    # Hybrid
    hybrid_model = HybridRecommender(cb_model, cf_model)
    
    return cb_model, cf_model, hybrid_model

@st.cache_resource
def get_popular_items(_interactions, recipes):
    """Get most popular recipes baseline."""
    item_stats = _interactions.groupby('recipe_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    item_stats.columns = ['recipe_id', 'avg_rating', 'count']
    
    C = item_stats['count'].mean()
    m = item_stats['avg_rating'].mean()
    item_stats['score'] = (item_stats['count'] * item_stats['avg_rating'] + C * m) / (item_stats['count'] + C)
    
    top_popular = item_stats.nlargest(50, 'score').merge(
        recipes[['id', 'name', 'description', 'minutes', 'n_ingredients']],
        left_on='recipe_id', right_on='id'
    )
    return top_popular


def recommend_from_text(user_input, cb_model, recipes, n=10):
    """
    Generate recommendations from text preferences using ContentBasedRecommender.
    We need to manually handle the 'user vector' creation since the class 
    methods expect user_id or interactions df usually.
    """
    tfidf = cb_model.tfidf_vectorizer
    item_features = cb_model.item_features
    
    # Transform user input
    user_tfidf = tfidf.transform([user_input])

    
    # Get dimensions
    text_dim = user_tfidf.shape[1]
    total_dim = item_features.shape[1]
    
    user_vector = np.zeros((1, total_dim))
    user_vector[0, :text_dim] = user_tfidf.toarray()[0] * 0.7
    
    # Compute similarity using the class's item features
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(user_vector, item_features)[0]
    
    top_indices = np.argsort(similarities)[::-1][:n]
    
    recs = []
    for idx in top_indices:
        rid = cb_model.idx_to_recipe_id[idx]
        row = recipes[recipes['id'] == rid].iloc[0]
        recs.append({
            'name': row['name'],
            'description': str(row['description'])[:200] if pd.notna(row['description']) else 'No description',
            'minutes': row['minutes'],
            'score': similarities[idx]
        })
        
    return pd.DataFrame(recs)

def recommend_from_session_ratings(rated_recipes, cb_model, recipes, n=10):
    """
    Generate recommendations from session ratings using ContentBasedRecommender structures.
    rated_recipes: dict {recipe_id: rating}
    """
    if not rated_recipes:
        return pd.DataFrame()
        
    item_features = cb_model.item_features
    recipe_id_to_idx = cb_model.recipe_id_to_idx
    idx_to_recipe_id = cb_model.idx_to_recipe_id
    
    weighted_sum = np.zeros(item_features.shape[1])
    total_weight = 0
    
    for rid, rating in rated_recipes.items():
        if rid in recipe_id_to_idx:
            idx = recipe_id_to_idx[rid]
            weighted_sum += item_features[idx] * rating
            total_weight += rating
            
    if total_weight == 0:
        return pd.DataFrame()
        
    user_profile = weighted_sum / total_weight
    
    # Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([user_profile], item_features)[0]
    
    # Mask already rated
    for rid in rated_recipes:
        if rid in recipe_id_to_idx:
            similarities[recipe_id_to_idx[rid]] = -1
            
    top_indices = np.argsort(similarities)[::-1][:n]
    
    recs = []
    for idx in top_indices:
        rid = idx_to_recipe_id[idx]
        row = recipes[recipes['id'] == rid].iloc[0]
        recs.append({
            'name': row['name'],
            'description': str(row['description'])[:200] if pd.notna(row['description']) else 'No description',
            'minutes': row['minutes'],
            'score': similarities[idx]
        })
        
    return pd.DataFrame(recs)

def main():
    st.markdown('<h1 class="main-header">Recipe Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #888;">Get personalized recipe recommendations based on your taste!</p>', unsafe_allow_html=True)
    
    # 1. Load Data
    with st.spinner("Loading recipe data..."):
        recipes, interactions = load_and_prep_data()
        cb_model, cf_model, hybrid_model = init_models(recipes, interactions)
        popular_items = get_popular_items(interactions, recipes)
    
    # Session State
    if 'rated_recipes' not in st.session_state:
        st.session_state.rated_recipes = {}
        
    # Sidebar
    st.sidebar.markdown("## Your Preferences")
    
    method = st.sidebar.radio(
        "Choose recommendation method:",
        ["Text Preferences", "Rate Recipes", "Popular Recipes"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Statistics")
    st.sidebar.metric("Total Recipes", f"{len(recipes):,}")
    st.sidebar.metric("Total Interactions", f"{len(interactions):,}")
    
    # --- METHOD 1: TEXT PREFERENCES ---
    if method == "Text Preferences":
        st.markdown("## Tell us what you like!")
        st.markdown("Describe your taste preferences - favorite ingredients, cuisines, cooking styles, dietary preferences, etc.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Your preferences:",
                placeholder="e.g., I love chocolate desserts, easy baking recipes, Italian pasta, healthy salads with chicken...",
                height=150
            )
            
            st.markdown("**Quick picks:**")
            quick_cols = st.columns(4)
            quick_options = {
                "Chocolate": "chocolate dessert sweet baking cocoa",
                "Healthy": "healthy salad vegetable low-calorie diet",
                "Italian": "italian pasta pizza tomato cheese mediterranean",
                "Spicy": "spicy hot chili pepper curry indian mexican",
                "Quick": "quick easy fast simple 30-minute breakfast",
                "Meat": "beef chicken meat protein grilled barbecue",
                "Vegan": "vegan vegetarian plant-based no-meat",
                "Dessert": "dessert cake cookies sweet baking sugar"
            }
            
            for i, (label, value) in enumerate(quick_options.items()):
                if quick_cols[i % 4].button(label, key=f"quick_{i}"):
                    user_input = value
                    
        with col2:
            n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
            
        if st.button("Get Recommendations", type="primary") or user_input:
            if user_input:
                with st.spinner("Finding perfect recipes for you..."):
                    recommendations = recommend_from_text(user_input, cb_model, recipes, n=n_recommendations)
                
                st.markdown("---")
                st.markdown("## Recommended Recipes for You")
                
                for i, row in recommendations.iterrows():
                    with st.container():
                        c1, c2, c3 = st.columns([3, 1, 1])
                        c1.markdown(f"### {i+1}. {row['name']}")
                        c1.markdown(f"*{row['description']}...*")
                        c2.metric("Minutes", int(row['minutes']) if pd.notna(row['minutes']) else 'N/A')
                        c3.metric("Match", f"{row['score']*100:.1f}%")
                        st.markdown("---")
            else:
                st.warning("Please enter your taste preferences!")

    # --- METHOD 2: RATE RECIPES ---
    elif method == "Rate Recipes":
        st.markdown("## Rate Some Recipes")
        st.markdown("Rate a few recipes to help us understand your taste. The more you rate, the better recommendations!")
        
        st.markdown("### Popular Recipes to Rate:")
        sample_recipes = popular_items.head(12)
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(sample_recipes.iterrows()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"**{row['name'][:40]}...**")
                    st.caption(f"Min: {int(row['minutes'])} | Ing: {int(row['n_ingredients'])}")
                    
                    current_rating = st.session_state.rated_recipes.get(row['id'], 0)
                    rating = st.slider(
                        "Rate:", 0, 5, current_rating, key=f"rate_{row['id']}"
                    )
                    
                    if rating > 0:
                        st.session_state.rated_recipes[row['id']] = rating
                        
        st.markdown("---")
        rated_count = len([r for r in st.session_state.rated_recipes.values() if r > 0])
        st.markdown(f"**You've rated {rated_count} recipes**")
        
        if st.button("Get Personalized Recommendations", type="primary"):
            active_ratings = {k: v for k, v in st.session_state.rated_recipes.items() if v > 0}
            
            if len(active_ratings) < 2:
                st.warning("Please rate at least 2 recipes to get recommendations!")
            else:
                with st.spinner("Building your taste profile..."):
                    # Use Content-Based logic for session-based recommendations (since user is anonymous/new)
                    recommendations = recommend_from_session_ratings(active_ratings, cb_model, recipes, n=10)
                
                st.markdown("---")
                st.markdown("## Personalized Recommendations")
                
                if not recommendations.empty:
                    for i, row in recommendations.iterrows():
                        with st.container():
                            c1, c2, c3 = st.columns([3, 1, 1])
                            c1.markdown(f"### {i+1}. {row['name']}")
                            c1.markdown(f"*{row['description']}...*")
                            c2.metric("Minutes", int(row['minutes']) if pd.notna(row['minutes']) else 'N/A')
                            c3.metric("Match", f"{row['score']*100:.1f}%")
                            st.markdown("---")
                else:
                    st.error("Could not generate recommendations.")
                    
        if st.button("Clear All Ratings"):
            st.session_state.rated_recipes = {}
            st.rerun()

    # --- METHOD 3: POPULAR RECIPES ---
    else:
        st.markdown("## Most Popular Recipes")
        st.markdown("These are the highest-rated recipes loved by our community!")
        
        for i, row in popular_items.head(15).iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(f"### {row['name']}")
                c2.metric("Rating", f"{row['avg_rating']:.1f}")
                c3.metric("Reviews", f"{int(row['count'])}")
                c4.metric("Minutes", int(row['minutes']) if pd.notna(row['minutes']) else 'N/A')
                st.markdown("---")
                
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>Built with Content-Based & Hybrid Recommendation Systems</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
