#  Telco Customer Churn Prediction & Analysis

##  Project Overview
Customer attrition (churn) is one of the most critical metrics for a telecommunications company. Acquiring new customers is significantly more expensive than retaining existing ones. The objective of this project is to build a machine learning pipeline to predict whether a customer will churn and to identify the key business drivers behind this behavior.

##  Technologies & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Logistic Regression, Data Preprocessing, Metrics)

##  Key Business Insights
Through exploratory data analysis (EDA) and model feature evaluation, several critical insights were discovered:
1. **Contract Type is Crucial:** Month-to-month contracts exhibit the highest churn rate (~42.7%), whereas one-year and two-year contracts have significantly lower churn rates (~11% and ~2.8%). 
2. **Fiber Optic & Electronic Check Risk:** Customers using Fiber Optic internet and Electronic Check payment methods show above-average churn rates.
3. **Tenure & Pricing Factor:** Churned customers have a significantly lower median tenure (~18 months) and higher monthly charges compared to retained customers, indicating early lifecycle vulnerability.

##  Model Performance
A **Logistic Regression** baseline model was built to predict customer churn. Since the dataset is inherently imbalanced (~26% churn rate), the model was configured with `class_weight='balanced'` to heavily penalize minority class misclassifications.
- **ROC-AUC Score:** `0.84`
- **Recall (Churn Class):** `0.79`

*The model successfully identifies the majority of potential churners, allowing the business to proactively target them with retention campaigns.*

##  How to Run the Project
1. Clone this repository: `git clone https://github.com/sultanesm/telco-customer-churn.git`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the Jupyter Notebook: `jupyter notebook TelcoCustomerChurn.ipynb`
