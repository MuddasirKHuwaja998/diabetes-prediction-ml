import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # DATA PREPROCESSING
    print("\n===> Data Preprocessing <===")
    print(f"\nOriginal shape: {df.shape}")
    
    # lets Drop features with weak correlation
    df = df.drop(columns=['SkinThickness', 'Insulin'])
    print(f"New shape after dropping weakly correlated features: {df.shape}")

    print("\nNow we will also remove rows from important features having zero values because in medical data if there will be zero then there are possibility of patient has not been diagnosed")
    print("Removing rows with zero values in Glucose, BloodPressure, BMI, and Age")
    df = df[df['Glucose'] != 0]
    df = df[df['BloodPressure'] != 0]
    df = df[df['BMI'] != 0]
    df = df[df['Age'] != 0]
    print(f"New shape after removing rows with zero values: {df.shape}")

    # SPLITTING THE DATASET INTO TRAINING AND TESTING SETS
    print("\n===> Splitting the Dataset into Training and Testing Sets <===")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                         y,
                                                         test_size=0.2,
                                                         random_state=42)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    print("EDA and Data Preprocessing Completed Successfully. We are now ready for modeling!")

    # Feature Scaling
    print("\n===> Feature Scaling <===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)      # Fit on train
    X_test_scaled = scaler.transform(X_test)            # Use same scaler on test
    print("Feature scaling applied to training and testing sets.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test