import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from processor import engineer_features

def train_abandonment_model():
    # Load Data
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, '..', 'data')
    
    df_events = pd.read_csv(os.path.join(data_dir, 'events.csv'))
    df_products = pd.read_csv(os.path.join(data_dir, 'products.csv'))

    # Engineer Features
    data = engineer_features(df_events, df_products)

    # Define Features (X) and Target (y)
    # We EXCLUDE user_id from X because the model cannot process strings like 'U0001'
    feature_cols = ['total_views', 'total_cart_adds', 'unique_categories', 'cart_to_view_ratio']
    X = data[feature_cols]
    y = data['is_abandoner']

    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save Model
    models_dir = os.path.join(base_path, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'abandonment_model.pkl'))
    
    print(f"✅ Model trained on {len(X_train)} samples and saved!")

if __name__ == "__main__":
    train_abandonment_model()