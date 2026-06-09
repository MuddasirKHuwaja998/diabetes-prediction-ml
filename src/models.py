from sklearn.metrics import f1_score, precision_score, recall_score , classification_report, confusion_matrix        
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_data
from eda import run_eda
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
import pickle
print ("Libraries imported successfully")

# Load and inspect data
df = run_eda(r'C:\Users\Mudda\Desktop\Machine learning2026\Diabaties\diabetes (1).csv')
# Preprocess data
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

#Modeling 
print("===> Modeling <===")
print("===> Modele : 1 (Logistic Regression) <===")
print("Let's start with Logistic Regression as a baseline model for classification.")
model = LogisticRegression(max_iter=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


#Evaluation
print("===> Model Evaluation of Logistic Regression <===")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"as data is imbalanced we should use F1 score as evaluation metric to penalise the model for misclassifying the minority class")
f1 = f1_score(y_test, y_pred)
print(f"\nF1 Score: {f1:.2f}")


print("\n===> Modele : 2 (Random Forest Classifier) <===")
model_2 = RandomForestClassifier(n_estimators= 100, random_state=42)# n_estimators is the number of trees in the forest, random_state is for reproducibility
model_2.fit(X_train_scaled, y_train)
y_pred_2 = model_2.predict(X_test_scaled)

#Evaluation of Random Forest Classifier
print("===> Model Evaluation of Random Forest Classifier <===")
print("Evaluation of Random Forest Classifier:")
print(classification_report(y_test, y_pred_2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_2))
print("F1 Score:")
f1_2 = f1_score(y_test, y_pred_2)
print(f"\nF1 Score: {f1_2:.2f}")

# Modeling with XGBOOST
print("\n===> Modele : 3 (XGBoost Classifier) <===")
model_3 = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
model_3.fit(X_train_scaled, y_train)
y_pred_3 = model_3.predict(X_test_scaled)


# Evaluation of XGBoost Classifier
print("===> Model Evaluation of XGBoost Classifier <===")
print("Evaluation of XGBoost Classifier:")
print(classification_report(y_test, y_pred_3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_3))
print("F1 Score:")
f1_3 = f1_score(y_test, y_pred_3)
print(f"\nF1 Score: {f1_3:.2f}")


# Final comparison
print("\n===> Final Model Comparison <===")
print(f"Logistic Regression F1: {f1:.2f}")
print(f"Random Forest F1: {f1_2:.2f}")
print(f"XGBoost F1: {f1_3:.2f}")


#Final Model Selection based on F1 Score
print("\n===> Final Model Selection based on F1 Score <===")
best = max([(f1, 'Logistic Regression'), (f1_2, 'Random Forest Classifier'), (f1_3, 'XGBoost Classifier')])
print(f"The best model based on F1 Score is: {best[1]} with an F1 Score of {best[0]:.2f}")


#lets save the best model for future use
print ("\n===> Saving the Best Model <===")
with open ('Diabaties_model.pkl', 'wb') as file :# here we are using 'wb' mode to write the model in binary format
    pickle.dump(model_3, file)
print("Best model saved successfully as 'Diabaties_model.pkl'")