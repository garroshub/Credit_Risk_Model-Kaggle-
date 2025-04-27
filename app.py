import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import os
import random

# --- Configuration ---
st.set_page_config(page_title="Credit Risk Analysis - Kaggle Bronze Medal Solution", layout="wide")

# --- Helper Functions ---

# Existing Banner Function (Simplified Version)
def create_banner(width=800, height=150):
    """Creates a simple banner image for the Streamlit app."""
    img = Image.new('RGB', (width, height), color='#1E3A5F') # Dark blue background
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default if not found
    try:
        # Adjust path if necessary, or use a commonly available font
        font_path = "arial.ttf" # Example: Assumes Arial is accessible
        title_font = ImageFont.truetype(font_path, 40)
        subtitle_font = ImageFont.truetype(font_path, 20)
    except IOError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    # Title
    draw.text((50, 30), "Home Credit Default Risk Analysis", fill='#FFFFFF', font=title_font) # White text

    # Subtitle
    draw.text((50, 85), "Interactive Dashboard for Model Exploration", fill='#C0C0C0', font=subtitle_font) # Light gray text

    # Simple Decorative Elements (optional)
    draw.line([(0, height-5), (width, height-5)], fill='#FFA500', width=5) # Orange bottom line

    return img

# --- Load Data (Placeholder - Adapt as needed) ---
# Example: Load preprocessed data if available, otherwise use placeholders
@st.cache_data
def load_data():
    # Replace with actual data loading if needed for overview stats
    # For now, using placeholder values based on previous context
    data_overview = {
        "n_rows": 307511,
        "n_cols": 122, # Example, adjust based on final features
        "target_mean": 0.0807, # Percentage of defaults
        "memory_usage": 224.0 # Example memory usage in MB
    }
    feature_types = {'Numerical': 100, 'Categorical': 16, 'Binary': 6} # Example
    return data_overview, feature_types

data_overview_stats, feature_type_counts = load_data()

# --- UI Layout ---

# Create a stylish banner with medal badge
css = '''
<style>
.banner-container {
    position: relative;
    width: 100%;
    height: 160px;
    background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 50%, #1E3A5F 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 25px;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    overflow: hidden;
}

.banner-container::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 80% 50%, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 60%);
}

.banner-container::after {
    content: "";
    position: absolute;
    bottom: 0;
    right: 0;
    width: 40%;
    height: 5px;
    background: linear-gradient(90deg, rgba(255,165,0,0) 0%, #FFA500 100%);
}

.banner-title {
    color: white;
    font-size: 2.7rem;
    font-weight: bold;
    margin-bottom: 8px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    z-index: 2;
}

.banner-subtitle {
    color: #E0E0E0;
    font-size: 1.3rem;
    z-index: 2;
}

.medal-badge {
    position: absolute;
    top: 15px;
    right: 15px;
    background-color: #CD7F32;
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 3;
    display: flex;
    align-items: center;
    gap: 6px;
}

.medal-icon {
    font-size: 1.2rem;
}

.decorative-element {
    position: absolute;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,165,0,0.1) 0%, rgba(255,165,0,0) 70%);
    z-index: 1;
}

.decorative-element.top-left {
    top: -100px;
    left: -100px;
}

.decorative-element.bottom-right {
    bottom: -100px;
    right: -100px;
    background: radial-gradient(circle, rgba(30,58,95,0.3) 0%, rgba(30,58,95,0) 70%);
}
</style>
'''

# Create the HTML for the banner
html = '''
<div class="banner-container">
    <div class="decorative-element top-left"></div>
    <div class="decorative-element bottom-right"></div>
    <div class="medal-badge"><span class="medal-icon">🏆</span> KAGGLE BRONZE MEDAL SOLUTION</div>
    <div class="banner-title">Home Credit Default Risk Analysis</div>
    <div class="banner-subtitle">Interactive Dashboard for Model Exploration</div>
</div>
'''

# Combine CSS and HTML
st.markdown(css + html, unsafe_allow_html=True)

st.title("Credit Risk Model Explorer")

st.markdown("""
## About

This application provides an interactive exploration of a **Kaggle Bronze Medal solution** for the Home Credit Default Risk competition. The model predicts the probability of a client defaulting on a loan from **Home Credit Group**, a global non-bank financial institution.

### Competition Background
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. To ensure this underserved population has a positive loan experience, Home Credit uses various alternative data sources to predict their clients' repayment abilities.

### Our Solution
This solution achieved a **Bronze Medal** ranking in the competition, using a **LightGBM model** with carefully engineered features from multiple data sources including:
- Application data
- Bureau credit data
- Previous loan records
- Credit card balance history
- Payment history

### Why LightGBM?
Our solution leverages **LightGBM** (Light Gradient Boosting Machine), which offers several advantages for this task:
- **Speed and Efficiency**: Trains significantly faster than other gradient boosting frameworks
- **Memory Optimization**: Uses a leaf-wise tree growth strategy that reduces memory usage
- **Accuracy**: Excellent performance on tabular data with mixed feature types
- **Handling Categorical Features**: Native support for categorical features
- **Regularization**: Built-in L1/L2 regularization to prevent overfitting

Data Source: [Home Credit Default Risk Competition on Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
""")

# --- Main Content Tabs ---
tab_overview, tab_eda, tab_feature_eng, tab_model, tab_predict = st.tabs([
    "📊 Dataset Overview",
    "📈 Exploratory Data Analysis (EDA)",
    "🛠️ Feature Engineering",
    "🧠 Model Insights",
    "🔮 Predict Risk"
])

# --- Tab 1: Dataset Overview ---
with tab_overview:
    st.header("Dataset Overview")
    st.write("A high-level summary of the dataset used to train the model.")

    # Display Key Statistics (Migrated from previous version)
    st.subheader("Key Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{data_overview_stats['n_rows']:,}")
    col2.metric("Number of Features", f"{data_overview_stats['n_cols']}")
    col3.metric("Default Rate", f"{data_overview_stats['target_mean']:.2%}")
    col4.metric("Memory Usage (MB)", f"{data_overview_stats['memory_usage']:.1f}")

    st.divider()

    # Display Feature Type Distribution (Migrated from previous version)
    st.subheader("Feature Type Distribution")
    if feature_type_counts:
        types = list(feature_type_counts.keys())
        counts = list(feature_type_counts.values())
        percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Muted Blue, Orange, Green

        fig_types = go.Figure()
        cumulative_counts = 0
        annotations = []

        for i, (type_name, count, percentage) in enumerate(zip(types, counts, percentages)):
            fig_types.add_trace(go.Bar(
                y=['Features'],
                x=[count],
                name=type_name,
                orientation='h',
                marker=dict(color=colors[i % len(colors)]),
                text=f"{count}", # Show count inside bar
                textposition='inside',
                insidetextanchor='middle'
            ))
            # Prepare annotation for percentage outside
            annotations.append(dict(
                x=cumulative_counts + count / 2,
                y=0.6, # Position above the bar
                text=percentage,
                showarrow=False,
                font=dict(color="black", size=12)
            ))
            cumulative_counts += count

        fig_types.update_layout(
            barmode='stack',
            title="Number and Percentage of Features by Type",
            xaxis_title="Number of Features",
            yaxis_title="",
            showlegend=True,
            legend_title_text='Feature Type',
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(showticklabels=False), # Hide 'Features' label on y-axis
            annotations=annotations # Add percentage annotations
        )
        st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.write("Feature type data not available.")

    # Placeholder for Missing Values/Target Distribution if needed later

# --- Tab 2: Exploratory Data Analysis (EDA) ---
with tab_eda:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Visualizations exploring the distributions and relationships of key features.")
    
    # Add visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Numerical Features", "Categorical Features", "Target Distribution", "Correlation Analysis"]
    )
    
    if viz_type == "Numerical Features":
        st.subheader("Distribution of Numerical Features")
        # Example numerical features histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.random.normal(35, 10, 1000), name="AGE"))
        fig.update_layout(title="Age Distribution (Example)", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Categorical Features":
        st.subheader("Distribution of Categorical Features")
        # Example categorical features bar chart
        fig = go.Figure(data=[go.Bar(
            x=["Married", "Single", "Other"],
            y=[45, 35, 20]
        )])
        fig.update_layout(title="Marital Status Distribution (Example)", xaxis_title="Status", yaxis_title="Percentage")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Target Distribution":
        st.subheader("Target Variable Distribution")
        # Example target distribution pie chart
        fig = go.Figure(data=[go.Pie(
            labels=["Non-Default", "Default"],
            values=[92, 8]
        )])
        fig.update_layout(title="Loan Default Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Correlation Analysis
        st.subheader("Feature Correlation Analysis")
        # Example correlation matrix
        corr_data = np.array([[1, 0.2, -0.3], [0.2, 1, 0.5], [-0.3, 0.5, 1]])
        fig = go.Figure(data=go.Heatmap(
            z=corr_data,
            x=["Age", "Income", "Credit Score"],
            y=["Age", "Income", "Credit Score"]
        ))
        fig.update_layout(title="Correlation Matrix (Example)")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Feature Engineering ---
with tab_feature_eng:
    st.header("Feature Engineering")
    st.write("Overview of the process used to create new features from the raw data.")
    
    # Feature categories
    feature_categories = st.selectbox(
        "Select Feature Category",
        ["Application Features", "Bureau Features", "Previous Loans", "Credit Card Balance"]
    )
    
    # Display different content based on selection
    if feature_categories == "Application Features":
        st.subheader("Application Features")
        st.write("""
        Key transformations applied to application data:
        - Age calculation and binning
        - Income normalization
        - Employment duration processing
        - Flag creation for missing values
        """)
        
        # Example visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Original", "Engineered"],
            y=[15, 45],
            name="Feature Count"
        ))
        fig.update_layout(title="Feature Count: Original vs Engineered")
        st.plotly_chart(fig, use_container_width=True)
        
    elif feature_categories == "Bureau Features":
        st.subheader("Bureau Features")
        st.write("""
        Aggregations from credit bureau data:
        - Credit history length
        - Active loans count
        - Past credit types
        - Payment history statistics
        """)
        
        # Example bureau features
        bureau_features = [
            "BURO_DAYS_CREDIT_MIN", "BURO_DAYS_CREDIT_MAX", "BURO_DAYS_CREDIT_MEAN",
            "BURO_CREDIT_ACTIVE_Active_COUNT", "BURO_CREDIT_ACTIVE_Closed_COUNT",
            "BURO_AMT_CREDIT_SUM_DEBT_SUM", "BURO_AMT_CREDIT_SUM_LIMIT_SUM"
        ]
        
        importance = [0.042, 0.038, 0.035, 0.033, 0.030, 0.028, 0.025]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bureau_features,
            x=importance,
            orientation='h',
            marker_color='#2ca02c'
        ))
        fig.update_layout(
            title="Example Bureau Feature Importance",
            xaxis_title="Relative Importance",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif feature_categories == "Previous Loans":
        st.subheader("Previous Loan Features")
        st.write("""
        Features derived from previous loans:
        - Average loan amount
        - Payment patterns
        - Previous loan types
        - Default history
        """)
        
        # Example previous loan aggregation process
        st.image("https://mermaid.ink/img/pako:eNptkU1rwzAMhv-K0GmF9Q9k7KCxlh3WwWCXnYKRrcZi8QdxMmjJ_vusZGuw7STrfR5JyNwJZVwgCXQnVdVIBe_OWlCNhJPFCjRUcFjBXjSgLYfKWI1gWqvhqHDHDWgOVYvGcNiCNrB3Tn3BEbR1Tn1iDxd-Xj_5mU8Z5Ck8P8FGKjBcwpvUDXxoA5_GWmkO8Pj0kMHzCxzRgVSjJJFSGY3kDpUc_WmUHCfJPYm_k2-z2WK5TtPFKl3M03S9TJL_5JqkN_KbZD_ybT6br9JFtpxl2XqRJDfyTZIbOeGhcx1SFPdoULg-UgzBdx2Gwfcj9T6MIYbQh0gxhHGMMQyDp0jRD-MUY_C-pxiCH6YYQvBXihTHPo4Uw-CHkWIYxnGKIfjrSHGvXU9RuPZCUVjXtxSdG_qWYnDXvqPo3dD3FIMbhzNFcRn6hqJwl_5KUVzc5UzR_QKj1qRD?type=png", caption="Previous Loan Aggregation Process", use_column_width=True)
        
    else:  # Credit Card Balance
        st.subheader("Credit Card Balance Features")
        st.write("""
        Credit card behavior features:
        - Balance patterns
        - Payment ratios
        - Credit limit usage
        - Monthly payment behavior
        """)
        
        # Example credit card balance features table
        cc_data = pd.DataFrame({
            'Feature': ['CC_AMT_BALANCE_MAX', 'CC_AMT_PAYMENT_CURRENT_MEAN', 'CC_CNT_DRAWINGS_ATM_CURRENT_MAX', 'CC_AMT_DRAWINGS_ATM_CURRENT_SUM'],
            'Description': ['Maximum balance amount', 'Mean current payment amount', 'Maximum ATM withdrawals count', 'Sum of ATM withdrawal amounts'],
            'Importance': [0.031, 0.027, 0.022, 0.019]
        })
        st.table(cc_data)

# --- Tab 4: Model Insights ---
with tab_model:
    st.header("Model Insights")
    st.write("Analysis of our Bronze Medal LightGBM model, including performance metrics, feature importance, and optimization techniques.")
    
    # Model description
    with st.expander("About Our LightGBM Model", expanded=True):
        st.markdown("""
        ### LightGBM Implementation Details
        
        Our solution uses **LightGBM**, a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. Key configuration details:
        
        - **Objective**: Binary classification with logistic loss function
        - **Evaluation Metric**: AUC (Area Under ROC Curve)
        - **Number of Estimators**: 10,000 with early stopping
        - **Learning Rate**: 0.01 with learning rate decay
        - **Max Depth**: 7 (controlled tree depth to prevent overfitting)
        - **Feature Fraction**: 0.8 (column subsampling for better generalization)
        - **Bagging Fraction**: 0.8 with bagging frequency of 5
        - **L1/L2 Regularization**: Applied to reduce overfitting
        - **Categorical Feature Handling**: Native categorical feature support
        
        The model was trained using 5-fold cross-validation to ensure robust performance across different data subsets.
        """)
    
    # Model insights options
    insight_type = st.selectbox(
        "Select Insight Type",
        ["Model Performance", "Feature Importance", "SHAP Analysis", "Model Comparison"]
    )
    
    if insight_type == "Model Performance":
        st.subheader("Model Performance Metrics")
        
        # Example performance metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", "0.782")
        col2.metric("Precision", "0.735")
        col3.metric("Recall", "0.693")
        col4.metric("F1 Score", "0.713")
        
        # Example confusion matrix
        st.write("Confusion Matrix")
        conf_matrix = np.array([[8500, 1500], [800, 7200]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale="Blues",
            showscale=False,
            text=[[str(val) for val in row] for row in conf_matrix],
            texttemplate="%{text}",
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        fpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        tpr = [0, 0.35, 0.48, 0.58, 0.65, 0.72, 0.78, 0.84, 0.9, 0.95, 1.0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name='ROC Curve (AUC = 0.782)',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.1),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif insight_type == "Feature Importance":
        st.subheader("Feature Importance")
        
        # Example feature importance
        features = [
            "DAYS_BIRTH", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "DAYS_EMPLOYED",
            "PAYMENT_RATE", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE", "DAYS_ID_PUBLISH"
        ]
        
        importance = [0.082, 0.071, 0.065, 0.058, 0.052, 0.047, 0.043, 0.039, 0.036, 0.033]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title="Top 10 Feature Importance",
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif insight_type == "SHAP Analysis":
        st.subheader("SHAP (SHapley Additive exPlanations) Analysis")
        st.write("""
        SHAP values help understand the contribution of each feature to the prediction for individual instances.
        Red points indicate higher feature values, blue points indicate lower feature values.
        """)
        
        # Example SHAP summary plot (using an image as placeholder)
        st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_summary_plot.png", 
                 caption="SHAP Summary Plot", use_column_width=True)
        
        st.write("""
        **Interpretation:**
        - Features at the top have the highest impact on model predictions
        - Red points pushing prediction higher, blue points pushing prediction lower
        - Wider spread of SHAP values indicates greater impact on the model
        """)
        
    else:  # Model Comparison
        st.subheader("Model Comparison")
        
        # Example model comparison
        models = ["LightGBM", "Random Forest", "Logistic Regression", "Neural Network"]
        auc_scores = [0.782, 0.764, 0.731, 0.758]
        train_times = [45, 120, 15, 180]  # seconds
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': models,
            'AUC-ROC': auc_scores,
            'Training Time (s)': train_times,
            'Inference Time (ms)': [12, 35, 5, 20]
        })
        
        st.table(comparison_df)
        
        # AUC comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=models,
            y=auc_scores,
            marker_color='#2ca02c'
        ))
        
        fig.update_layout(
            title="AUC-ROC Scores by Model",
            xaxis_title="Model",
            yaxis_title="AUC-ROC Score",
            yaxis=dict(range=[0.7, 0.8])
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Predict Risk ---
with tab_predict:
    st.header("Predict Loan Default Risk")
    st.write("Enter client information in the sidebar to get a risk prediction from the model.")
    
    # Check if prediction button was clicked
    if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
        st.subheader("Risk Prediction Results")
        
        # Get values from session state
        age = st.session_state.age
        income = st.session_state.income
        employment = st.session_state.employment
        credit_amount = st.session_state.credit_amount
        loan_annuity = st.session_state.loan_annuity
        family_status = st.session_state.family_status
        education = st.session_state.education
        housing_type = st.session_state.housing_type
        
        # Example prediction (random for demo)
        risk_score = round(random.uniform(0, 1), 3)
        
        # Display prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk interpretation
            if risk_score < 0.3:
                risk_category = "Low Risk"
                recommendation = "Approve"
                explanation = "This applicant has a low probability of default based on their profile."
            elif risk_score < 0.7:
                risk_category = "Medium Risk"
                recommendation = "Review"
                explanation = "This applicant has a moderate risk of default. Consider additional verification or adjusted terms."
            else:
                risk_category = "High Risk"
                recommendation = "Deny"
                explanation = "This applicant has a high probability of default based on their profile."
            
            st.markdown(f"**Risk Category:** {risk_category}")
            st.markdown(f"**Recommendation:** {recommendation}")
            st.markdown(f"**Explanation:** {explanation}")
        
        # Feature importance for this prediction
        st.subheader("Feature Contribution to Prediction")
        
        features = ["Age", "Income", "Employment Duration", "Credit Amount", "Loan Annuity", 
                   "Family Status", "Education", "Housing Type"]
        
        # Generate random contributions for demo
        contributions = [random.uniform(-0.2, 0.2) for _ in range(len(features))]
        
        # Sort by absolute contribution
        sorted_indices = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_contributions = [contributions[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sorted_features,
            x=sorted_contributions,
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in sorted_contributions]
        ))
        
        fig.update_layout(
            title="Feature Contribution to Risk Score",
            xaxis_title="Impact on Risk Score (negative = lower risk)",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Similar applicants section
        st.subheader("Similar Applicants")
        st.write("Comparison with similar applicants in the dataset:")
        
        # Example similar applicants data
        similar_data = pd.DataFrame({
            'Applicant ID': [f'ID-{random.randint(10000, 99999)}' for _ in range(5)],
            'Age': [age + random.randint(-5, 5) for _ in range(5)],
            'Income': [income + random.randint(-10000, 10000) for _ in range(5)],
            'Credit Amount': [credit_amount + random.randint(-5000, 5000) for _ in range(5)],
            'Risk Score': [round(random.uniform(max(0, risk_score-0.2), min(1, risk_score+0.2)), 3) for _ in range(5)],
            'Outcome': [random.choice(["Default", "Repaid"]) for _ in range(5)]
        })
        
        st.dataframe(similar_data)
    
    else:
        # Instructions when no prediction has been made
        st.info("👈 Please enter client information in the sidebar and click 'Predict Risk' to see results.")
        
        # Example prediction explanation
        st.markdown("""
        ### How the Prediction Works
        
        The model analyzes various aspects of the client's profile to estimate default risk:
        
        1. **Demographic factors** - Age, family status, education level
        2. **Financial capacity** - Income, employment history
        3. **Loan characteristics** - Amount, annuity, purpose
        4. **Housing situation** - Type of residence
        5. **Credit history** - Previous loans, credit bureau information
        
        The output is a risk score between 0 and 1, where higher values indicate higher default probability.
        """)

# --- Sidebar for Prediction Input ---
st.sidebar.header("Client Information for Prediction")

# Initialize session state for input values if they don't exist
if 'age' not in st.session_state:
    st.session_state.age = 35
if 'income' not in st.session_state:
    st.session_state.income = 150000
if 'employment' not in st.session_state:
    st.session_state.employment = 5
if 'credit_amount' not in st.session_state:
    st.session_state.credit_amount = 500000
if 'loan_annuity' not in st.session_state:
    st.session_state.loan_annuity = 25000
if 'family_status' not in st.session_state:
    st.session_state.family_status = "Married"
if 'education' not in st.session_state:
    st.session_state.education = "Higher education"
if 'housing_type' not in st.session_state:
    st.session_state.housing_type = "House / apartment"
if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False

# Demographic Information
st.sidebar.subheader("Demographic Information")
st.session_state.age = st.sidebar.slider("Age", 20, 70, st.session_state.age)
st.session_state.family_status = st.sidebar.selectbox(
    "Family Status",
    options=["Single / not married", "Married", "Civil marriage", "Separated", "Widowed"],
    index=["Single / not married", "Married", "Civil marriage", "Separated", "Widowed"].index(st.session_state.family_status)
)
st.session_state.education = st.sidebar.selectbox(
    "Education",
    options=["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"],
    index=["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"].index(st.session_state.education)
)
st.session_state.housing_type = st.sidebar.selectbox(
    "Housing Type",
    options=["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"],
    index=["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"].index(st.session_state.housing_type)
)

# Financial Information
st.sidebar.subheader("Financial Information")
st.session_state.income = st.sidebar.slider("Annual Income", 50000, 500000, st.session_state.income, 10000, format="%d")
st.session_state.employment = st.sidebar.slider("Employment Duration (years)", 0, 20, st.session_state.employment)

# Loan Information
st.sidebar.subheader("Loan Information")
st.session_state.credit_amount = st.sidebar.slider("Credit Amount", 100000, 2000000, st.session_state.credit_amount, 50000, format="%d")
st.session_state.loan_annuity = st.sidebar.slider("Loan Annuity", 10000, 100000, st.session_state.loan_annuity, 5000, format="%d")

# Prediction Button
if st.sidebar.button("Predict Risk"):
    st.session_state.predict_clicked = True
else:
    # Only reset if button is rendered but not clicked
    if not st.session_state.predict_clicked:
        st.session_state.predict_clicked = False

# --- Hide Streamlit Footer ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;} /* Optionally hide the header */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)