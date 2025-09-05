# Loan-Approval-Prediction-Data-Engineering-ML-Pipeline
This project builds an end-to-end pipeline to predict loan approval decisions using a combination of data engineering and machine learning best practices.

Key contents:
A reproducible ETL notebook that loads cleaned loan datasets into MySQL: creates tables, enforces schema, inserts batched records, runs basic validation queries and exports query results for downstream ML training.

Data merging & cleaning — merges applicant_info, financial_info and loan_info, handles missing values, converts Dependents ('3+') to numeric, and encodes the target (Loan_Status).

Feature engineering — creates stable features such as Total_Income, EMI_proxy, and Loan_to_Income; applies log transforms for skewed financial fields.

Preprocessing pipeline — a ColumnTransformer handles median imputation and scaling for numeric features and OneHotEncoder for categoricals; pipeline is wrapped with models to guarantee reproducible preprocessing at inference.

Modeling & evaluation — baseline Logistic Regression and Random Forest models with stratified train/test split and k-fold cross-validation; metrics reported include accuracy, precision, recall, F1 and confusion matrices; feature importance and coefficient plots included for interpretability.

Hyperparameter tuning — GridSearchCV/RandomizedSearchCV examples to reduce overfitting via regularization (LR) and tree complexity limits (RF).

Deployment — saves the full preprocessing + model pipeline as model.pkl for use in a Streamlit app or API.
