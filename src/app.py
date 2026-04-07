import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from processor import engineer_features, get_similarity_matrix

# --- 1. Page Config & Branding ---
st.set_page_config(page_title="Nano-Retail AI Assistant", layout="wide")

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

X_input = [[current_user['total_views'], current_user['total_cart_adds'], 
            current_user['unique_categories'], current_user['cart_to_view_ratio']]]
risk_proba = model.predict_proba(X_input)[0][1] if model else 0.0

# --- 5. Main UI ---
st.title("🛒 AI Smart E-Commerce Dashboard")
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
    st.subheader("Collaborative Filtering Recommendations")
    ui_matrix, sim_df = get_similarity_matrix(df_events)
    
    if selected_id in sim_df.index:
        # Find similar users and items they liked 
        similar_users = sim_df[selected_id].sort_values(ascending=False)[1:4].index.tolist()
        current_user_items = ui_matrix.loc[selected_id]
        recs = []
        
        for user in similar_users:
            top_items = ui_matrix.loc[user][ui_matrix.loc[user] > 2].index.tolist()
            recs.extend([i for i in top_items if current_user_items[i] == 0])
        
        final_recs = list(set(recs))[:3]
        if final_recs:
            rec_df = df_products[df_products['product_id'].isin(final_recs)]
            cols = st.columns(len(rec_df))
            for i, (idx, row) in enumerate(rec_df.iterrows()):
                with cols[i]:
                    st.write(f"**{row['product_name']}**")
                    st.button(f"Buy RM{row['price']}", key=f"rec_{i}")
        else:
            st.write("Browsing more items will help AI improve recommendations.")

# --- 6. AI Explainability & Merchant Strategies ---
st.markdown("---")
with st.expander("💡 AI Explainability & Actionable Insights"):
    st.write(f"**Analysis for User {selected_id}:**")
    
    if risk_proba > 0.6:
        st.error(f"High Abandonment Risk ({risk_proba*100:.1f}%) detected.")
        
        # Strategy 1: Price Sensitive Abandoner
        if current_user['cart_to_view_ratio'] > 0.7:
            st.subheader("🎯 Strategy: Price-Incentive Conversion")
            st.write("Interpretation: User is adding almost everything they see to the cart but not checking out. This usually indicates price comparison behavior.")
            st.info("**Merchant Action:** Trigger a limited-time '10% Discount Code' or 'Free Shipping' popup immediately.")
            
        # Strategy 2: Indecisive/Comparison Shopper
        elif current_user['unique_categories'] > 5:
            st.subheader("🎯 Strategy: Curation & Focus")
            st.write("Interpretation: User is browsing too many different categories. They are likely overwhelmed or 'window shopping'.")
            st.info("**Merchant Action:** Send a 'Best Sellers' email or show a 'Most Popular in [Top Category]' side-widget to narrow their focus.")
            
        # Strategy 3: Low Engagement Abandoner
        else:
            st.subheader("🎯 Strategy: Urgency & Scarcity")
            st.write("Interpretation: Low session depth despite items in cart. User might be losing interest.")
            st.info("**Merchant Action:** Display a 'Low Stock' alert or '3 people have this in their cart' notification to create urgency.")

    else:
        st.success("Stable behavior profile. Strategy: Standard retargeting and loyalty rewards.")