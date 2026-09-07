# **Credit Risk Prediction \- Loan Default Classification**

## **Project Overview**

This project aims to build a robust Machine Learning model to predict credit risk and classify potential loan defaults. It was developed as part of the **Samsung Innovation Campus | Life Makers Foundation \- AI 801 & 802** program.  
The final solution not only includes predictive modeling but also features an interactive web application for real-time credit risk assessment.

## **Machine Learning Pipeline**

Our team followed a comprehensive end-to-end data science lifecycle:

> 1. **Data Preprocessing:** Handled missing values, removed duplicates, corrected suspicious entries, and applied appropriate encoding and scaling techniques.  
> 2. **Exploratory Data Analysis (EDA):** Analyzed data distributions, feature correlations, and key relationships to uncover underlying risk patterns.  
> 3. **Feature Engineering:** Crafted meaningful new features to enhance model predictive power.  
> 4. **Class Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the default and non-default classes.  
> 5. **Model Building & Experimentation:** Trained and compared multiple models including Logistic Regression, Random Forest, and XGBoost.  
> 6. **Model Evaluation:** Evaluated models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.

## **Model Performance & Selection**

After rigorous experimentation, **XGBoost** emerged as the best-performing model, particularly in identifying potential defaults, achieving the following metrics:

> * **Accuracy:** 92.4%  
> * **Recall:** 80.5%

## **Technologies & Libraries Used**

> * **Programming Language:** Python  
> * **Data Manipulation & Analysis:** pandas  
> * **Machine Learning:** scikit-learn, XGBoost  
> * **Web Deployment:** Streamlit

## **Repository Structure**

├── app.py                           \# Streamlit application for real-time deployment  
├── Credit\_Risk.ipynb                \# Main Jupyter Notebook containing EDA and model training  
├── credit\_risk\_dataset.csv          \# The raw dataset used for the project  
├── model\_metrics.csv                \# Extracted metrics for model comparisons  
├── models/                          \# Directory containing serialized models and preprocessors (.pkl)  
│   ├── income\_threshold.pkl  
│   ├── logistic\_smote.pkl  
│   ├── logistic\_weighted.pkl  
│   ├── preprocessor\_unscaled.pkl  
│   ├── xgboost\_normal.pkl  
│   ├── xgboost\_smote.pkl  
│   └── xgboost\_weighted.pkl  
└── Second Project Guidlines.pdf     \# Project requirements and guidelines

## **How to Run the Application**

> 1. Clone this repository to your local machine:  
>    git clone https://github.com/YourUsername/Your-Repo-Name.git  
> 2. Navigate to the project directory:  
>    cd Your-Repo-Name  
> 3. Install the required dependencies (Make sure you have Streamlit, XGBoost, scikit-learn, and pandas installed).  
> 4. Run the Streamlit app:  
>    streamlit run app.py

## **Team Members**

Proud of what we built together\!

> * **\[Your Name\]**  
> * **Eng. Toka Abdelaziz**  
> * **Eng. Esraa EL-Tohamy**  
> * **Eng. Ammar Yasser**

## **Acknowledgments**

Special thanks to our facilitator **Eng. Mariam El Gazzar** for her continuous support and valuable feedback, and to **Eng. Abdul Rahman Abdelalem** for his guidance throughout the project at Samsung Innovation Campus.
