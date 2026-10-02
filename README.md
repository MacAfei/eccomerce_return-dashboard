# eccomerce_return-dashboard
# 📦 E-Commerce Product Return Risk Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Altair](https://img.shields.io/badge/Altair-6.2-4682B4?style=for-the-badge)](https://altair-viz.github.io/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

An end-to-end Machine Learning intelligence dashboard designed to predict product return probability **before shipment dispatch**, evaluate reverse-logistics financial exposure, and trigger proactive operational interventions to reduce returns and prevent Return-to-Origin (RTO) losses.

## 🌟 Key Highlights

- **Pre-Dispatch Return Prediction:** Predicts return probability using a trained **Gradient Boosting** regression pipeline trained on customer profiles, transactional attributes, and order economics.
- **Financial Risk Exposure:** Automatically calculates **Expected Loss ($)** based on unit pricing, ordered quantity, and predicted return likelihood.
- **Three-Tier Operational Matrix:** Dynamically classifies orders into **Low Risk** (<30%), **Medium Risk** (30%–70%), and **High Risk** (≥70%), providing tailored fulfillment guidelines and risk factors.
- **Batch CSV Scoring:** Upload multi-order CSV files or evaluate built-in cohorts with one-click bulk scoring and downloadable results.
- **Population Analytics:** Comprehensive exploratory data analysis across 2,007 evaluated transactions, analyzing return trends across categories, price bands, and payment methods.

## 🖥️ Dashboard Architecture & Modules

The application is structured into four primary modules:

 E-Commerce Return Risk Dashboard
├── 🎯 Tab 1: Single Order Predictor
│   ├── Interactive order parameter configuration (Customer age, tenure, category, pricing, discount, etc.)
│   ├── Quick-fill preset dropdown with verified test samples
│   ├── Real-time Risk Classification Gauge & Status Badge
│   ├── Financial metrics: Order Value & Expected Return Exposure
│   └── Prescriptive Fulfillment Recommendations & Risk Factor breakdown
├── 📊 Tab 2: Dataset Analytics & Overview
│   ├── Population KPIs (n = 2,007 orders, 17.3% actual return rate)
│   ├── Risk tier distribution bar charts
│   ├── Category-wise return rate analysis (Clothing, Electronics, Beauty, etc.)
│   ├── Payment method risk profile & pricing distribution box plots
│   └── Interactive filterable order table
├── 📁 Tab 3: Batch CSV Predictor
│   ├── File uploader supporting `.csv` files
│   ├── Fallback loader for built-in 40-order evaluation dataset
│   ├── Real-time parallel row inference
│   ├── Batch summary metrics (average probability, low vs flagged breakdown)
│   └── Export button for scored dataset with predictions (`ecommerce_orders_scored.csv`)
└── 🧠 Tab 4: Model Performance & Framework
    ├── Evaluation metrics: Accuracy (82.46%), Precision (40.91%), ROC-AUC (0.624)
    ├── Machine learning pipeline details (OneHotEncoder + GradientBoostingRegressor)
    └── Three-tier operational intervention matrix
```

## 🚦 Risk Stratification & Intervention Matrix

| Risk Tier | Probability Range | Badge Color | Primary Objective | Operational Protocol |
| :--- | :--- | :--- | :--- | :--- |
| 🟢 **Low Risk** | `P < 0.30` | Emerald Green | Automated Fulfillment | Standard automated dispatch; automated delivery tracking and loyalty rewards. |
| 🟠 **Medium Risk** | `0.30 ≤ P < 0.70` | Amber Orange | Proactive Engagement | Automated SMS / WhatsApp confirmation verifying sizing, color, or specifications; exchange assistance. |
| 🔴 **High Risk** | `P ≥ 0.70` | Crimson Red | Pre-Shipment Audit | Hold order for customer intent and address verification; pre-dispatch quality checks; COD-to-prepaid conversion incentives. |

---

## 🧬 Machine Learning Pipeline

### Feature Set (9 Features)
1. **`customer_age`** (*int*): Customer age in years.
2. **`customer_tenure_days_at_order`** (*int*): Days since customer registered on the platform.
3. **`product_category`** (*categorical*): Product category (`Clothing`, `Electronics`, `Home`, `Beauty`, `Books`, etc.).
4. **`unit_price`** (*float*): Price per unit in USD ($).
5. **`quantity`** (*int*): Number of units purchased in the order.
6. **`discount_pct`** (*float*): Promotional discount applied (0% - 100%).
7. **`payment_method`** (*categorical*): Payment channel (`Credit Card`, `Debit Card`, `PayPal`).
8. **`shipping_cost`** (*float*): Incurred shipping charge in USD ($).
9. **`orders_before_this_one`** (*int*): Total count of prior orders placed by this customer.

### Pipeline Definition
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['product_category', 'payment_method']),
        ('num', 'passthrough', ['customer_age', 'customer_tenure_days_at_order', 'unit_price', 'quantity', 'discount_pct', 'shipping_cost', 'orders_before_this_one'])
    ]
)

regressor = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
```

---

## 📂 Repository Structure

```
eccomerce_return_dashboard/
├── app.py                             # Main Streamlit web application & UI layout
├── model_utils.py                     # Inference logic, normalization mappings, and risk formatters
├── train_model.py                     # Model training & export script
├── ecommerce_return_model.pkl         # Serialized Scikit-Learn Gradient Boosting pipeline
├── ecommerce_sample_40.csv            # 40 verified sample orders with ground truth predictions
├── ecommerce_return_predictions (1).csv # Full evaluation dataset (2,007 orders)
├── pyrightconfig.json                 # Type checker & language server configuration
├── .gitignore                         # Exclusions for virtual environments and cache
└── .vscode/
    └── settings.json                  # IDE interpreter and path configurations
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- Git (optional, for cloning).

### 2. Setup Virtual Environment
In the project root folder, create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required packages:
```bash
pip install streamlit pandas numpy scikit-learn altair joblib
```

### 4. Run the Streamlit Dashboard
Launch the web interface using Streamlit:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at:
👉 **`http://localhost:8501/`**

---

## 🔁 Retraining the Model

To retrain the Gradient Boosting model or rebuild the serialized pipeline artifact:
```bash
python train_model.py
```
This script reads `ecommerce_sample_40.csv` (and optionally `ecommerce_return_predictions (1).csv`), fits the preprocessing and estimator pipeline, evaluates maximum error, and updates `ecommerce_return_model.pkl`.

---

## 🛠️ Operational Recommendations Formula

The expected monetary risk for any order is computed dynamically as:
$$\text{Order Value} = \text{Unit Price} \times \text{Quantity}$$
$$\text{Expected Loss Exposure} = \text{Order Value} \times P(\text{Return})$$

Where $P(\text{Return}) \in [0, 1]$ is the predicted probability generated by the Gradient Boosting pipeline.

---

## 📄 License & Academic Attribution
Developed as part of an academic and enterprise-grade **E-Commerce Return Risk Prediction & Analytics** initiative. Designed for operations managers, supply chain analysts, and e-commerce merchants.
