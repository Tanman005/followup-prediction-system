# Follow-Up Appointment Prediction System

# Overview

This project predicts whether a patient is likely to require a follow-up appointment based on medical and appointment-related data. It applies machine learning techniques to analyze patterns in patient behavior and health conditions.



# Problem Statement

In healthcare systems, missed follow-ups or unnecessary appointments lead to inefficiencies and poor patient outcomes.
This project aims to assist in decision-making by predicting the need for follow-up appointments using historical data.


# Dataset

 Source: Medical Appointment No Shows dataset
 Features include:

Age
Gender
Hypertension
Diabetes
Alcoholism
SMS reminders
Appointment scheduling details



# Technologies Used

 Python
 Pandas
 Scikit-learn
 Matplotlib
 Seaborn
 Streamlit



# Machine Learning Models

The following models were implemented and compared:

 Logistic Regression
 Logistic Regression (Balanced)
 Decision Tree
 Random Forest



# Results

 Logistic Regression achieved the highest accuracy (~71%)
 Random Forest provided a better balance between precision and recall
 Class imbalance affected model performance, which was addressed using class weighting



# Visualizations

 Model comparison graph
 Confusion matrix heatmap



# Streamlit Web App

A user-friendly interface was developed using Streamlit where users can:

 Input patient details
 Predict follow-up requirement instantly



# How to Run the Project

# 1. Clone the repository


git clone <your-github-link>
cd BYOP


# 2. Install dependencies


pip install -r requirements.txt


# 3. Run the ML pipeline


python test.py


# 4. Run the web app


streamlit run app/app.py




# Project Structure


BYOP/
│── app/
│   └── app.py
│── data/
│   └── appointments.csv
│── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── visualization.py
│── test.py
│── README.md
│── requirements.txt




#  Key Learnings

Importance of data preprocessing and feature engineering
 Handling class imbalance in datasets
 Comparing multiple machine learning models
 Building end-to-end ML applications
 Creating interactive UI using Streamlit



# Future Improvements

 Hyperparameter tuning for better accuracy
 Deployment on cloud platforms
 Integration with real hospital systems
 Use of deep learning models



# Author

Tanman Raj
23BAI11341
B.Tech (AIML)
