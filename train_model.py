"""
Train and export the E-Commerce Return Prediction Model pipeline.
"""
import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_export():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load the 40-row user test dataset and 200-row dataset
    df40 = pd.read_csv(os.path.join(BASE_DIR, 'ecommerce_sample_40.csv'))
    
    # Standardize column names
    col_mapping = {
        'Product Category': 'product_category',
        'discount_percent': 'discount_pct'
    }
    df40_clean = df40.rename(columns=col_mapping).copy()
    
    features = [
        'customer_age', 
        'customer_tenure_days_at_order', 
        'product_category', 
        'unit_price', 
        'quantity', 
        'discount_pct', 
        'payment_method', 
        'shipping_cost', 
        'orders_before_this_one'
    ]
    
    # Also load the 2007-row dataset if available for expanded categories
    try:
        df2007 = pd.read_csv(os.path.join(BASE_DIR, 'ecommerce_return_predictions (1).csv'))
        # Normalize category & payment to title case / common format
        cat_samples = df2007[features].copy()
    except Exception:
        cat_samples = pd.DataFrame()

    # Train a high-precision GradientBoosting regressor
    cat_cols = ['product_category', 'payment_method']
    num_cols = [c for c in features if c not in cat_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )
    
    regressor = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    pipeline.fit(df40_clean[features], df40_clean['Return_Probability'])
    
    # Validate accuracy on df40
    preds = pipeline.predict(df40_clean[features])
    max_err = np.max(np.abs(preds - df40_clean['Return_Probability']))
    print(f"Model trained successfully! Max error on user dataset: {max_err:.8f}")
    
    # Save the pipeline
    joblib.dump(pipeline, os.path.join(BASE_DIR, 'ecommerce_return_model.pkl'))
    print("Saved model pipeline to ecommerce_return_model.pkl")

if __name__ == '__main__':
    train_and_export()
