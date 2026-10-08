# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import altair as alt
import model_utils
def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if pd.notna(val) else default
    except Exception:
        return default

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if pd.notna(val) else default
    except Exception:
        return default

def _safe_str(val, default: str = "") -> str:
    if pd.isna(val) or val is None:
        return default
    return str(val)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="E-Commerce Return Risk Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Clean, Premium UI ---
st.markdown("""
<style>
    /* Global styling */
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
    }
    .risk-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .risk-med {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .risk-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .prediction-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .factor-item {
        padding: 0.4rem 0;
        color: #334155;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="main-header">📦 E-Commerce Product Return Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pre-dispatch Return Prediction Engine & Reverse-Logistics Risk Intelligence</div>', unsafe_allow_html=True)

# Load data assets
sample_df = model_utils.load_sample_dataset()
full_df = model_utils.load_full_scored_dataset()

# Top level tabs
tab_pred, tab_analytics, tab_batch, tab_model = st.tabs([
    "🎯 Single Order Predictor", 
    "📊 Dataset Analytics & Overview", 
    "📁 Batch CSV Predictor", 
    "🧠 Model Performance & Framework"
])

# ==============================================================================
# TAB 1: SINGLE ORDER RISK PREDICTOR
# ==============================================================================
with tab_pred:
    st.subheader("1. Configure Order & Customer Profile")
    
    # Preset sample selector
    col_preset1, col_preset2 = st.columns([3, 1])
    with col_preset1:
        preset_options = ["Custom Input"]
        if not sample_df.empty:
            for idx, (_, r) in enumerate(sample_df.iterrows()):
                preset_options.append(
                    f"Sample #{idx+1}: {r['Product Category']} | Age {int(r['customer_age'])} | "
                    f"${float(r['unit_price']):.2f} | {r['Risk_Level']} ({float(r['Return_Probability'])*100:.1f}%)"
                )
        selected_preset = st.selectbox(
            "Quick-Fill with Verified Test Sample:", 
            preset_options, 
            index=0,
            help="Select any sample from the verified evaluation dataset to auto-populate the form and test exact predictions."
        )

    # Defaults for form
    default_age = 43
    default_tenure = 676
    default_category = "Beauty"
    default_price = 38.24
    default_qty = 4
    default_discount = 18.6
    default_payment = "Debit Card"
    default_shipping = 10.11
    default_orders = 8

    # If user selected a preset sample, load values
    if selected_preset != "Custom Input" and not sample_df.empty:
        idx = int(selected_preset.split(":")[0].replace("Sample #", "")) - 1
        row = sample_df.iloc[idx]
        default_age = _safe_int(row.get('customer_age'), default_age)
        default_tenure = _safe_int(row.get('customer_tenure_days_at_order'), default_tenure)
        default_category = _safe_str(row.get('Product Category'), default_category)
        default_price = _safe_float(row.get('unit_price'), default_price)
        default_qty = _safe_int(row.get('quantity'), default_qty)
        default_discount = _safe_float(row.get('discount_percent'), default_discount)
        default_payment = _safe_str(row.get('payment_method'), default_payment)
        default_shipping = _safe_float(row.get('shipping_cost'), default_shipping)
        default_orders = _safe_int(row.get('orders_before_this_one'), default_orders)

    categories_list = ["Beauty", "Clothing", "Electronics", "Home & Kitchen", "Books", "Sports", "Toys", "Food"]
    if default_category not in categories_list:
        if default_category == "Home":
            default_category = "Home & Kitchen"
        else:
            categories_list.append(default_category)

    payment_list = ["Debit Card", "Credit Card", "PayPal", "Cash on Delivery", "Bank Transfer", "UPI"]
    if default_payment not in payment_list:
        payment_list.append(default_payment)

    # Input Form Layout
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 👤 Customer Profile")
        customer_age = st.number_input("Customer Age (Years)", min_value=18, max_value=100, value=default_age, step=1)
        customer_tenure = st.number_input("Customer Tenure (Days with Platform)", min_value=0, max_value=3000, value=default_tenure, step=10)
        previous_orders = st.number_input("Orders Before This One", min_value=0, max_value=100, value=default_orders, step=1)

    with c2:
        st.markdown("##### 🛍️ Product & Pricing")
        product_category = st.selectbox(
            "Product Category", 
            categories_list, 
            index=categories_list.index(default_category) if default_category in categories_list else 0
        )
        unit_price = st.number_input("Unit Price ($)", min_value=1.0, max_value=5000.0, value=round(default_price, 2), step=1.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=50, value=default_qty, step=1)
        discount_pct = st.slider("Discount (%)", min_value=0.0, max_value=100.0, value=round(default_discount, 1), step=0.5)

    with c3:
        st.markdown("##### 🚚 Payment & Fulfillment")
        payment_method = st.selectbox(
            "Payment Method", 
            payment_list, 
            index=payment_list.index(default_payment) if default_payment in payment_list else 0
        )
        shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, max_value=100.0, value=round(default_shipping, 2), step=0.5)
        st.caption(f"Gross Order Value: **${unit_price * quantity:.2f}**")

    st.write("")
    predict_clicked = st.button("🚀 Predict Return Risk", type="primary", use_container_width=True)

    # Run Prediction
    if predict_clicked or selected_preset != "Custom Input":
        result = model_utils.predict_single_order(
            customer_age=customer_age,
            customer_tenure_days_at_order=customer_tenure,
            product_category=product_category,
            unit_price=unit_price,
            quantity=quantity,
            discount_pct=discount_pct,
            payment_method=payment_method,
            shipping_cost=shipping_cost,
            orders_before_this_one=previous_orders
        )

        st.divider()
        st.subheader("2. Real-Time Risk Intelligence")

        # Top Metric Cards
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            st.metric("Return Probability", result["probability_percent"])
            st.progress(float(result["return_probability"]))

        with col_res2:
            risk_class = (
                "risk-low" if result["risk_level"] == "Low Risk" 
                else ("risk-med" if result["risk_level"] == "Medium Risk" else "risk-high")
            )
            st.markdown(f"**Risk Classification**")
            st.markdown(f'<div class="risk-badge {risk_class}">{result["risk_level"]}</div>', unsafe_allow_html=True)

        with col_res3:
            st.metric("Predicted Outcome", result["status_label"])

        with col_res4:
            st.metric("Expected Return Exposure", f"${result['expected_loss']:.2f}")

        # Operational Details Card
        factors_html = "".join([f"<li class='factor-item'>{f}</li>" for f in result['risk_factors']])
        recs_html = "".join([f"<li class='factor-item'>{r}</li>" for r in result['recommendations']])

        st.markdown(f"""
        <div class="prediction-box" style="border-left: 6px solid {result['badge_color']};">
            <h4 style="margin-top:0; color:{result['badge_color']};">
                📋 Recommendation: {result['action_headline']}
            </h4>
            <div style="margin-top:0.8rem;">
                <strong>Key Contributing Risk Drivers:</strong>
                <ul>
                    {factors_html}
                </ul>
            </div>
            <div style="margin-top:0.8rem;">
                <strong>Actionable Dispatch & Logistics Guidelines:</strong>
                <ul>
                    {recs_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: DATASET ANALYTICS & OVERVIEW
# ==============================================================================
with tab_analytics:
    st.subheader("Evaluated Order Risk Analytics (n = 2,007)")
    st.markdown("Comprehensive distribution and segment analytics computed from the trained model's evaluated cohort.")

    # High level KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Evaluated Orders", "2,007", help="Stratified test evaluation split")
    with k2:
        st.metric("Actual Return Rate", "17.3%", help="348 returned out of 2,007 orders")
    with k3:
        st.metric("Mean Predicted Probability", "17.4%", help="Model population average probability")
    with k4:
        st.metric("Low Risk Share", "91.9%", "1,844 orders (< 30% prob)")

    st.write("")

    # Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 🛡️ Orders by Risk Classification")
        risk_summary = pd.DataFrame({
            "Risk Tier": ["Low Risk (<30%)", "Medium Risk (30-70%)", "High Risk (≥70%)"],
            "Order Count": [1844, 158, 5],
            "Percentage": ["91.9%", "7.9%", "0.2%"],
            "Color": ["#10b981", "#f59e0b", "#ef4444"]
        })
        
        bar_chart = alt.Chart(risk_summary).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Risk Tier:N", sort=None, title="Risk Level"),
            y=alt.Y("Order Count:Q", title="Number of Orders"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Risk Tier", "Order Count", "Percentage"]
        ).properties(height=280)
        st.altair_chart(bar_chart, use_container_width=True)

    with chart_col2:
        st.markdown("##### 🏷️ Category Return Risk Profile")
        cat_summary = pd.DataFrame({
            "Category": ["Clothing", "Electronics", "Automotive", "Home & Kitchen", "Beauty", "Food", "Books"],
            "Avg Return Prob (%)": [25.3, 20.2, 18.5, 16.8, 16.8, 11.7, 10.5],
            "Actual Return Rate (%)": [26.3, 20.9, 18.1, 16.5, 17.0, 6.4, 10.0]
        })
        
        cat_chart = alt.Chart(cat_summary).mark_bar().encode(
            x=alt.X("Category:N", sort="-y", title="Product Category"),
            y=alt.Y("Avg Return Prob (%):Q", title="Return Probability (%)"),
            color=alt.value("#3b82f6"),
            tooltip=["Category", "Avg Return Prob (%)", "Actual Return Rate (%)"]
        ).properties(height=280)
        st.altair_chart(cat_chart, use_container_width=True)

    c_row2_1, c_row2_2 = st.columns(2)

    with c_row2_1:
        st.markdown("##### 🏷️ Return Rate by Discount Band")
        discount_bands = pd.DataFrame({
            "Discount Band": ["0% Discount", "1 - 5%", "6 - 10%", "11 - 15%", "> 20% Discount"],
            "Return Rate (%)": [15.1, 16.2, 18.1, 28.6, 34.5]
        })
        disc_chart = alt.Chart(discount_bands).mark_line(point=True, color="#8b5cf6").encode(
            x=alt.X("Discount Band:N", sort=None, title="Discount Tier"),
            y=alt.Y("Return Rate (%):Q", title="Actual Return Rate (%)"),
            tooltip=["Discount Band", "Return Rate (%)"]
        ).properties(height=250)
        st.altair_chart(disc_chart, use_container_width=True)

    with c_row2_2:
        st.markdown("##### 💳 Payment Method Distribution & Risk")
        pm_summary = pd.DataFrame({
            "Payment Method": ["Debit Card", "Credit Card", "PayPal", "Bank Transfer", "Cash on Delivery"],
            "Avg Probability (%)": [18.9, 18.4, 15.7, 15.2, 12.0]
        })
        pm_chart = alt.Chart(pm_summary).mark_bar(color="#06b6d4").encode(
            x=alt.X("Payment Method:N", sort="-y", title="Payment Method"),
            y=alt.Y("Avg Probability (%):Q", title="Avg Return Probability (%)"),
            tooltip=["Payment Method", "Avg Probability (%)"]
        ).properties(height=250)
        st.altair_chart(pm_chart, use_container_width=True)

    st.write("")
    st.markdown("##### 🔍 Interactive Scored Orders Table")
    if not sample_df.empty:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_filter = st.multiselect("Filter by Risk Level", ["Low Risk", "Medium Risk", "High Risk"], default=["Low Risk", "Medium Risk"])
        with filter_col2:
            cat_filter = st.multiselect("Filter by Category", sample_df['Product Category'].unique().tolist(), default=sample_df['Product Category'].unique().tolist()[:3])

        filtered_table = sample_df[
            (sample_df['Risk_Level'].isin(risk_filter)) &
            (sample_df['Product Category'].isin(cat_filter))
        ]
        st.dataframe(filtered_table, use_container_width=True, height=260)


# ==============================================================================
# TAB 3: BATCH CSV PREDICTOR & EXPORTER
# ==============================================================================
with tab_batch:
    st.subheader("Batch Order Scoring Engine")
    st.markdown("Upload any CSV order manifest to generate return probabilities, risk tiers, and fulfillment recommendations.")

    batch_source = st.radio(
        "Choose Batch Data Source:",
        ["Built-in Evaluation Dataset (40 Orders)", "Upload Custom CSV File"],
        horizontal=True
    )

    df_to_score = None
    if batch_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_to_score = pd.read_csv(uploaded_file)
                st.success(f"Uploaded CSV loaded successfully ({len(df_to_score)} rows).")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    else:
        if not sample_df.empty:
            df_to_score = sample_df.copy()
            st.info("Loaded built-in 40-order evaluation dataset. Ready for batch scoring.")

    if df_to_score is not None:
        st.markdown("##### Preview Input Data")
        st.dataframe(df_to_score.head(5), use_container_width=True)

        if st.button("⚡ Score Batch Orders", type="primary"):
            with st.spinner("Executing predictions across batch..."):
                scored_rows = []
                for _, row in df_to_score.iterrows():
                    # Handle flexible column names
                    age = _safe_int(row.get('customer_age'), 30)
                    tenure = _safe_int(row.get('customer_tenure_days_at_order'), 365)
                    cat_val = row.get('Product Category') if 'Product Category' in row else row.get('product_category')
                    cat = _safe_str(cat_val, 'Beauty')
                    price = _safe_float(row.get('unit_price'), 50.0)
                    qty = _safe_int(row.get('quantity'), 1)
                    disc_val = row.get('discount_percent') if 'discount_percent' in row else row.get('discount_pct')
                    disc = _safe_float(disc_val, 10.0)
                    pm = _safe_str(row.get('payment_method'), 'Credit Card')
                    ship = _safe_float(row.get('shipping_cost'), 5.0)
                    orders_before = _safe_int(row.get('orders_before_this_one'), 1)

                    pred_res = model_utils.predict_single_order(
                        customer_age=age,
                        customer_tenure_days_at_order=tenure,
                        product_category=cat,
                        unit_price=price,
                        quantity=qty,
                        discount_pct=disc,
                        payment_method=pm,
                        shipping_cost=ship,
                        orders_before_this_one=orders_before
                    )
                    scored_rows.append({
                        "Predicted_Probability": pred_res["return_probability"],
                        "Predicted_Risk_Level": pred_res["risk_level"],
                        "Predicted_Status": pred_res["status_label"],
                        "Expected_Loss_USD": round(pred_res["expected_loss"], 2),
                        "Fulfillment_Action": pred_res["action_headline"]
                    })

                scored_df = pd.DataFrame(scored_rows)
                results_df = pd.concat([df_to_score.reset_index(drop=True), scored_df], axis=1)

                st.success("Batch scoring completed successfully!")
                
                # Summary metrics of batch
                total_orders = len(scored_rows)
                avg_prob = float(np.mean([r["Predicted_Probability"] for r in scored_rows]))
                low_cnt = sum(1 for r in scored_rows if r["Predicted_Risk_Level"] == "Low Risk")
                med_high_cnt = total_orders - low_cnt

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total Scored", total_orders)
                with b2:
                    st.metric("Average Return Prob", f"{avg_prob * 100:.1f}%")
                with b3:
                    st.metric("Low Risk Orders", f"{low_cnt} ({low_cnt / total_orders * 100:.1f}%)")
                with b4:
                    st.metric("Flagged for Review", f"{med_high_cnt} orders")

                st.write("")
                st.dataframe(results_df, use_container_width=True, height=300)

                csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored CSV with Predictions",
                    data=csv_bytes,
                    file_name="ecommerce_orders_scored.csv",
                    mime="text/csv",
                    type="primary"
                )


# ==============================================================================
# TAB 4: MODEL PERFORMANCE & ARCHITECTURE
# ==============================================================================
with tab_model:
    st.subheader("Machine Learning Methodology & Validation Metrics")
    st.markdown("Performance benchmarks and validation framework as documented in the B.Tech project review.")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", "82.46%", help="Proportion of total predictions that were correct")
    with m2:
        st.metric("Precision", "40.91%", help="Precision of positive return predictions")
    with m3:
        st.metric("Recall", "2.59%", help="Sensitivity across class-imbalanced return events")
    with m4:
        st.metric("F1-Score", "4.86%", help="Harmonic mean of precision and recall")
    with m5:
        st.metric("ROC-AUC", "0.624", help="Area under the Receiver Operating Characteristic curve")

    st.write("")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("##### 🏗️ Pipeline Architecture")
        st.markdown("""
        1. **Source Data**: ~10,035 e-commerce transactions (`ecommerce_orders.csv`).
        2. **Feature Preprocessing**:
           - **Categorical Encoding**: One-Hot Encoding for `product_category` and `payment_method`.
           - **Numerical Scaling**: Standardization & passthrough for continuous features (`customer_age`, `tenure`, `unit_price`, `quantity`, `discount_pct`, `shipping_cost`, `orders_before_this_one`).
        3. **Models Evaluated**:
           - **Logistic Regression**: Statistical baseline model.
           - **Random Forest**: Ensemble decision-tree classifier.
           - **XGBoost / Gradient Boosting**: Boosted trees for non-linear feature interactions and high ranking precision.
        4. **Evaluation Split**: 80:20 stratified split (`n=2,007` test transactions).
        """)

    with col_arch2:
        st.markdown("##### 🚦 Risk Stratification Matrix")
        st.markdown("""
        | Risk Tier | Probability Range | Action Headline | Fulfillment Protocol |
        | :--- | :--- | :--- | :--- |
        | 🟢 **Low Risk** | `P < 0.30` | Standard Dispatch | Automated express shipping |
        | 🟠 **Medium Risk** | `0.30 ≤ P < 0.70` | Proactive Engagement | Size/color SMS check & tracking |
        | 🔴 **High Risk** | `P ≥ 0.70` | Pre-Shipment Audit | Manual verification & packing audit |
        """)
        st.info("Risk thresholds were established to prioritize inventory availability and minimize return shipping overhead without inducing fulfillment friction on high-confidence orders.")

# --- Footer ---
st.divider()
st.caption("E-Commerce Return Risk Dashboard • Machine Learning Project • Developed with Streamlit")
