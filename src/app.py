import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from processor import engineer_features, get_similarity_matrix

# --- 1. Page Config & Branding ---
st.set_page_config(page_title="Smart E-Commerce AI Assistant", layout="wide")

# --- 2. Data & Model Loading ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, '..', 'data')
    try:
        u = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        p = pd.read_csv(os.path.join(data_dir, 'products.csv'))
        e = pd.read_csv(os.path.join(data_dir, 'events.csv'))
        u['user_id'], e['user_id'] = u['user_id'].astype(str), e['user_id'].astype(str)
        return u, p, e
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'abandonment_model.pkl')
    return joblib.load(path) if os.path.exists(path) else None

df_users, df_products, df_events = load_data()
model = load_model()

# --- 3. Sidebar Simulation ---
with st.sidebar:
    if df_users is not None:
        user_list = sorted(df_users['user_id'].unique())
        selected_id = st.selectbox("Select User ID", user_list)
    st.write("AI Model:", "🟢 Random Forest Online" if model else "🔴 Offline")

# --- 4. Prediction Logic ---
user_features = engineer_features(df_events, df_products)
current_user = user_features[user_features['user_id'] == selected_id].iloc[0]

# 1. Define the exact feature names used during training
feature_cols = ['total_views', 'total_cart_adds', 'unique_categories', 'cart_to_view_ratio']

# 2. Convert your input into a DataFrame with those names
X_input = pd.DataFrame([[
    current_user['total_views'], 
    current_user['total_cart_adds'], 
    current_user['unique_categories'], 
    current_user['cart_to_view_ratio']
]], columns=feature_cols)

# 3. Use the DataFrame for prediction
risk_proba = model.predict_proba(X_input)[0][1] if model else 0.0

# --- 5. Main UI ---
st.title("🛒 Smart E-Commerce AI Assistant Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Views", int(current_user['total_views']))
m2.metric("Cart Adds", int(current_user['total_cart_adds']))
m3.metric("Categories", int(current_user['unique_categories']))
m4.metric("Abandonment Risk", f"{risk_proba*100:.1f}%", delta="Confidence")

st.markdown("---")
tab1, tab2 = st.tabs(["📊 Behavior Analytics", "🎯 Next Best Product"])

with tab1:
    c1, c2 = st.columns(2)
    u_ev = df_events[df_events['user_id'] == selected_id]
    with c1:
        st.plotly_chart(px.pie(u_ev, names='event_type', title="User Intent"), width='stretch')
    with c2:
        merged = u_ev.merge(df_products, on='product_id')
        if not merged.empty:
            st.plotly_chart(px.bar(merged['category'].value_counts().reset_index(), x='category', y='count'), width='stretch')

with tab2:
    st.subheader("🎯 Collaborative Filtering Recommendations")
    ui_matrix, sim_df = get_similarity_matrix(df_events)
    
    if selected_id in sim_df.index:
        # 1. Find the top 3 most similar users (Neighbors)
        # We exclude the first one because it's the user themselves
        similar_users = sim_df[selected_id].sort_values(ascending=False)[1:4].index.tolist()
        sim_scores = sim_df[selected_id].sort_values(ascending=False)[1:4].values.tolist()
        
        current_user_items = ui_matrix.loc[selected_id]
        recs = []
        
        for user in similar_users:
            # Recommend products the similar user 'rated' highly (>2 means at least a Cart or Purchase)
            top_items = ui_matrix.loc[user][ui_matrix.loc[user] > 2].index.tolist()
            recs.extend([i for i in top_items if current_user_items[i] == 0])
        
        final_recs = list(set(recs))[:3]
        
        if final_recs:
            # --- The "Why" Explanation Section ---
            st.info(f"**AI Insight:** These products are recommended because your behavior matches 3 other 'Neighbor' shoppers with a **{(sum(sim_scores)/3)*100:.1f}% similarity score**.")
            
            rec_df = df_products[df_products['product_id'].isin(final_recs)]
            cols = st.columns(len(rec_df))
            
            for i, (idx, row) in enumerate(rec_df.iterrows()):
                with cols[i]:
                    st.write(f"**{row['product_name']}**")
                    st.caption(f"Category: {row['category']}")
                    st.button(f"Buy RM{row['price']}", key=f"rec_{i}")
                    
            # Explaining the Collaborative Logic
            with st.expander("🔍 How does this recommendation work?"):
                st.write("""
                **User-to-User Collaborative Filtering:**
                - Our AI calculated a **Similarity Matrix** using Cosine Similarity.
                - We identified users who have similar 'event profiles' to yours (similar views and carts).
                - We then looked for products those 'Neighbors' purchased that you haven't seen yet.
                - **The Goal:** To suggest high-intent items based on community trends rather than just your own history.
                """)
        else:
            st.write("AI is still building your neighborhood profile. Try viewing more items!")

# --- 6. AI Explainability & Actionable Insights ---
st.markdown("---")
with st.expander("💡 AI Explainability"):
    st.write(f"### Deep Analysis for User: {selected_id}")
    
    # --- Part A: Explaining the Score ---
    st.subheader("1. Risk Score Derivation")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Model Input Features:**")
        # Display the actual raw numbers the model is looking at
        st.write(f"- **Engagement Depth:** {int(current_user['total_views'])} views")
        st.write(f"- **Intent Signal:** {int(current_user['total_cart_adds'])} items added to cart")
        st.write(f"- **Interest Breadth:** {int(current_user['unique_categories'])} unique categories explored")
        st.write(f"- **Conversion Tension:** {current_user['cart_to_view_ratio']:.2f} (Cart-to-View Ratio)")

    # --- Part B: Strategy Decision Tree ---
    st.markdown("---")
    st.subheader("2. Strategic Decision Engine")
    
    if risk_proba > 0.6:
        st.error(f"⚠️ High Abandonment Risk: {risk_proba*100:.1f}% AI Confidence")
        
        # We define why we chose the specific strategy
        if current_user['cart_to_view_ratio'] > 0.8:
            st.subheader("🎯 Selection: Price-Incentive Strategy")
            st.write("**Decision Logic:** Since the user's Cart-to-View ratio is extremely high (>0.8), the AI interprets this as 'Price Sensitivity'. The user wants the items but is hesitant at the final price point.")
            st.success("**Merchant Action:** Automated 10% 'Checkout Now' discount triggered.")
            
        elif current_user['unique_categories'] > 5:
            st.subheader("🎯 Selection: Curation & Focus Strategy")
            st.write("**Decision Logic:** The user has explored over 5 categories. High category diversity often leads to 'Decision Fatigue'. They are lost in the catalog.")
            st.success("**Merchant Action:** Show 'Top 3 Recommendations' in their most-viewed category to narrow choice.")
            
        else:
            st.subheader("🎯 Selection: Urgency & Scarcity Strategy")
            st.write("**Decision Logic:** The user has items in the cart but low overall session engagement. They are likely to go 'cold' soon.")
            st.success("**Merchant Action:** Display 'Limited Stock' or 'Flash Sale' timer to create immediate FOMO.")

    else:
        st.success("✅ Stable Behavior Profile")
        st.write("**Decision Logic:** Behavior matches typical 'Healthy Browsing' patterns. No aggressive intervention needed.")
        st.info("**Merchant Action:** Maintain standard loyalty retargeting.")