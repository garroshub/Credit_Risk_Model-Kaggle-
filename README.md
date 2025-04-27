# Credit Risk Model - Kaggle Bronze Medal Solution

![Kaggle Bronze Medal](https://img.shields.io/badge/Kaggle-Bronze%20Medal-CD7F32)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-FF4B4B)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)

An interactive Streamlit dashboard showcasing a Bronze Medal solution for the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk). This application allows users to explore the dataset, understand feature engineering techniques, analyze model performance, and make risk predictions.

## 🏆 About the Competition

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. To ensure this underserved population has a positive loan experience, Home Credit uses various alternative data sources to predict their clients' repayment abilities.

The challenge was to build a model that predicts how capable each applicant is of repaying a loan, ensuring that clients capable of repayment are not rejected while loans are not given to clients who are likely to default.

## 📊 Application Features

The dashboard is organized into five main sections:

1. **Dataset Overview**: High-level summary of the dataset with key statistics
2. **Exploratory Data Analysis (EDA)**: Interactive visualizations exploring feature distributions and relationships
3. **Feature Engineering**: Detailed explanation of feature transformations and creation
4. **Model Insights**: Analysis of model performance, feature importance, and SHAP values
5. **Predict Risk**: Interactive interface to input client information and get risk predictions

## 🔍 Model Details

Our solution uses **LightGBM**, a gradient boosting framework that offers several advantages:

- **Speed and Efficiency**: Trains significantly faster than other gradient boosting frameworks
- **Memory Optimization**: Uses a leaf-wise tree growth strategy that reduces memory usage
- **Accuracy**: Excellent performance on tabular data with mixed feature types
- **Handling Categorical Features**: Native support for categorical features
- **Regularization**: Built-in L1/L2 regularization to prevent overfitting

Key configuration details:
- **Objective**: Binary classification with logistic loss function
- **Evaluation Metric**: AUC (Area Under ROC Curve)
- **Number of Estimators**: 10,000 with early stopping
- **Learning Rate**: 0.01 with learning rate decay
- **Max Depth**: 7 (controlled tree depth to prevent overfitting)
- **Feature Fraction**: 0.8 (column subsampling for better generalization)
- **Bagging Fraction**: 0.8 with bagging frequency of 5
- **L1/L2 Regularization**: Applied to reduce overfitting
- **Categorical Feature Handling**: Native categorical feature support

## 🚀 Installation and Usage

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/garroshub/Credit_Risk_Model-Kaggle-.git
   cd Credit_Risk_Model_Kaggle
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Deployment

The app can be deployed on Streamlit Cloud:

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy the app from your forked repository

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Home Credit Group](https://www.homecredit.net/) for providing the dataset
- [Kaggle](https://www.kaggle.com/) for hosting the competition
- The open-source community for the amazing tools and libraries

## 📧 Contact

For any questions or feedback, please open an issue on GitHub or contact the repository owner.
