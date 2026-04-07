import pandas as pd
import numpy as np

def engineer_features(df_events, df_products):
    """Transforms raw logs into features for abandonment prediction."""
    df_events['user_id'] = df_events['user_id'].astype(str)
    
    # Aggregate behavior per user [cite: 80, 81]
    user_behavior = df_events.groupby('user_id').agg(
        total_views=('event_type', lambda x: (x == 'view').sum()),
        total_cart_adds=('event_type', lambda x: (x == 'cart').sum()),
        total_purchases=('event_type', lambda x: (x == 'purchase').sum()),
    ).reset_index()

    # Calculate Category Diversity
    event_details = df_events.merge(df_products, on='product_id')
    diversity = event_details.groupby('user_id')['category'].nunique().reset_index()
    diversity.columns = ['user_id', 'unique_categories']
    
    user_behavior = user_behavior.merge(diversity, on='user_id', how='left').fillna(0)

    # Risk Scoring Logic
    user_behavior['cart_to_view_ratio'] = user_behavior['total_cart_adds'] / (user_behavior['total_views'] + 1)
    user_behavior['is_abandoner'] = ((user_behavior['total_cart_adds'] > 0) & 
                                    (user_behavior['total_purchases'] == 0)).astype(int)

    return user_behavior

def get_similarity_matrix(df_events):
    """Builds the User-Item matrix for Collaborative Filtering."""
    # Assign weights to actions to simulate 'ratings'
    event_weights = {'purchase': 5, 'cart': 3, 'view': 1}
    df_events['rating'] = df_events['event_type'].map(event_weights)
    
    # Create Matrix: Rows = Users, Columns = Products 
    ui_matrix = df_events.pivot_table(
        index='user_id', 
        columns='product_id', 
        values='rating', 
        fill_value=0
    )
    
    # Calculate Cosine Similarity between user vectors 
    from sklearn.metrics.pairwise import cosine_similarity
    user_sim = cosine_similarity(ui_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=ui_matrix.index, columns=ui_matrix.index)
    
    return ui_matrix, user_sim_df