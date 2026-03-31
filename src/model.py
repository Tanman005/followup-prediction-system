import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.visualization import plot_model_accuracy, plot_confusion_matrix


def train_models(df):
    # Drop unnecessary columns
    df = df.drop(['ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)

    # Features & target
    X = df.drop('FollowUp', axis=1)
    y = df['FollowUp']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # 🔹 Logistic Regression (Normal)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)

    # 🔹 Logistic Regression (Balanced)
    lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_bal.fit(X_train, y_train)
    y_pred_lr_bal = lr_bal.predict(X_test)
    results['Logistic (Balanced)'] = accuracy_score(y_test, y_pred_lr_bal)

    # 🔹 Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)

    # 🔹 Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, y_pred_rf)

    # ✅ Print results (only once)
    print("\nModel Accuracies:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")

    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf))

    # 📊 Plot accuracy comparison
    plot_model_accuracy(results)

    # 🔥 Plot confusion matrix (THIS WAS MISSING/WRONG)
    plot_confusion_matrix(y_test, y_pred_rf)