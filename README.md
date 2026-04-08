🛍️ Smart E-Commerce AI Assistant

Team Name: Team P (League 2)

Project Objective: Transforming Online Retail with Predictive Analytics

Problem Statement 2: AI for Smart E-Commerce

📌 Project Overview

Smart E-Commerce AI Assistant is a dual-engine AI prototype designed to empower Malaysian businesses especially MSMEs (Micro, Small, Medium, Enterprise) by tackling two major retail challenges: Cart Abandonment and Lack of Personalization.

    1. Abandonment Predictor: Uses a Random Forest Classifier to identify high-risk shoppers in real-time.

    2. Personalization Engine: Implements User-to-User Collaborative Filtering to recommend the "Next Best Product" based on similar user personas.

    3. Actionable Insights: Bridges the gap between "AI Predictions" and "Merchant Actions" through heuristic-based business strategies.

🛠️ Tech Stack

    - Language: Python 3.10+

    - Framework: Streamlit (Web UI)

    - ML Libraries: Scikit-Learn, Pandas, NumPy

    - Visualization: Plotly, Matplotlib

    - Model Persistence: Joblib

🚀 Getting Started

    1. Prerequisites

    Ensure you have Python installed. It is highly recommended to use a virtual environment:

    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate

    2. Installation

    Install the required dependencies:

    pip install -r requirements.txt

    3. Training the Model

    Before running the app, you must generate the trained Random Forest model:

    python src/train_model.py

    This will create a models/abandonment_model.pkl file.

    4. Running the Application

    Launch the Streamlit dashboard:

    streamlit run src/app.py

🧠 AI Methodology

    - Feature Engineering: Raw event strings (e.g., 'U000001') are transformed into behavior metrics like "Cart-to-View Ratio" and "Category Diversity".

    - Classification: A Random Forest model provides a probability score (0.0 to 1.0) indicating abandonment risk.

    - Similarity: Cosine Similarity is applied to a User-Item matrix to find "nearest neighbor" preferences for recommendations.

    - Explainability: The system uses logic-based transparency to explain why a specific marketing strategy was selected.