
import pandas as pd
import numpy as np
import ast
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir='../data'):
    """Load raw datasets from data directory."""
    print("Loading datasets...")
    interactions_path = os.path.join(data_dir, 'RAW_interactions.csv')
    recipes_path = os.path.join(data_dir, 'RAW_recipes.csv')
    
    if not os.path.exists(interactions_path) or not os.path.exists(recipes_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
        
    interactions = pd.read_csv(interactions_path)
    recipes = pd.read_csv(recipes_path)
    print(f"Loaded {len(interactions):,} interactions and {len(recipes):,} recipes")
    return interactions, recipes

def clean_interactions(interactions):
    """Clean interactions data."""
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING: INTERACTIONS")
    print("=" * 50)
    
    # Drop rows with missing values
    initial_len = len(interactions)
    interactions_clean = interactions.dropna(subset=['user_id', 'recipe_id', 'rating'])
    
    # Fill missing reviews
    if 'review' in interactions_clean.columns:
        interactions_clean['review'] = interactions_clean['review'].fillna('')
        
    print(f"Rows before cleaning: {initial_len:,}")
    print(f"Rows after cleaning: {len(interactions_clean):,}")
    print(f"Rows removed: {initial_len - len(interactions_clean):,}")
    
    # Handle duplicates
    duplicates = interactions_clean.duplicated(subset=['user_id', 'recipe_id']).sum()
    print(f"Duplicate user-item pairs: {duplicates:,}")
    
    interactions_clean = interactions_clean.drop_duplicates(
        subset=['user_id', 'recipe_id'], 
        keep='last'
    )
    print(f"Rows after removing duplicates: {len(interactions_clean):,}")
    
    return interactions_clean

def scale_ratings(interactions):
    """Scale ratings to 1-5 range."""
    print("\n--- Rating Scaling ---")
    min_rating = interactions['rating'].min()
    max_rating = interactions['rating'].max()
    
    print(f"Original rating range: {min_rating} - {max_rating}")
    
    if min_rating == 0 and max_rating == 5:
        # Scale 0-5 to 1-5
        interactions['rating'] = interactions['rating'].apply(lambda x: max(1, x))
        print("Scaled ratings from 0-5 to 1-5 (converted 0s to 1s)")
    elif max_rating > 5 or min_rating < 0:
        # Min-Max scaling to 1-5
        interactions['rating'] = 1 + 4 * (interactions['rating'] - min_rating) / (max_rating - min_rating)
        print(f"Applied Min-Max scaling to 1-5 range")
    else:
        print("Ratings already in 1-5 range")
        
    print(f"New rating range: {interactions['rating'].min()} - {interactions['rating'].max()}")
    return interactions

def preprocess_recipes(recipes):
    """Preprocess recipe data (parse lists, create combined text)."""
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING: RECIPES")
    print("=" * 50)
    
    def safe_eval(x):
        """Safely evaluate string representations of lists"""
        try:
            if pd.isna(x):
                return []
            return ast.literal_eval(x) if isinstance(x, str) else x
        except:
            return []

    print("Parsing list columns...")
    # Parse list columns
    recipes['ingredients_list'] = recipes['ingredients'].apply(safe_eval)
    recipes['tags_list'] = recipes['tags'].apply(safe_eval)
    
    print("Creating combined text for TF-IDF...")
    # Create combined text for TF-IDF
    recipes['combined_text'] = (
        recipes['name'].fillna('') + ' ' +
        recipes['description'].fillna('') + ' ' +
        recipes['ingredients'].apply(lambda x: ' '.join(safe_eval(x)) if pd.notna(x) else '') + ' ' +
        recipes['tags'].apply(lambda x: ' '.join(safe_eval(x)) if pd.notna(x) else '')
    )
    
    return recipes

def save_processed_data(interactions, recipes, data_dir='../data'):
    """Save processed datasets."""
    print("\n" + "=" * 50)
    print("SAVING PROCESSED DATA")
    print("=" * 50)
    
    interactions_path = os.path.join(data_dir, 'interactions_cleaned.csv')
    recipes_path = os.path.join(data_dir, 'recipes_processed.csv')
    
    interactions.to_csv(interactions_path, index=False)
    # We save recipes primarily to keep the 'combined_text' if needed later, 
    # though usually we re-process to ensure consistency. 
    # For this refactor, saving it helps avoid re-processing time if we wanted to reload.
    # However, 'recipes_processed.csv' wasn't explicitly in the original list, 
    # but 'interactions_cleaned.csv' was. We'll save 'interactions_cleaned.csv'.
    
    print(f"✓ Cleaned interaction data saved to '{interactions_path}'")
    # recipes.to_csv(recipes_path, index=False) # Optional, can uncomment if needed

if __name__ == "__main__":
    # --- 1. Load Data ---
    # Assuming script is run from 'code/' directory, data is in '../data/'
    try:
        interactions, recipes = load_data()
        
        # --- 2. Clean Interactions ---
        interactions = clean_interactions(interactions)
        interactions = scale_ratings(interactions)
        
        # --- 3. Preprocess Recipes ---
        # Note: Preprocessing recipes (text combination) is needed for Content-Based.
        # We can either do it here and save, or do it on the fly in content_based.py.
        # The notebook did it as part of the pipeline. 
        # For efficiency, we can keep it here if we saved it, but since we modify 'recipes' df 
        # which is passed to other modules, we'll verify if we need to save it.
        # The notebook saved 'interactions_cleaned.csv' but re-processed recipes text in Part 2.
        # We will create the combined text column here just to verify it works, 
        # but the actual logic usually resides where it's used or we export it.
        # In this refactor, I will leave the heavy text processing in this script 
        # but ensure 'interactions_cleaned.csv' is the main artifact.
        
        recipes = preprocess_recipes(recipes)
        
        # --- 4. Save Data ---
        save_processed_data(interactions, recipes)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")