import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully")

def run_eda(csv_path):
    print("===> EDA (Exploratory Data Analysis) <===")
    df = pd.read_csv(csv_path)
    print("Dataset Loaded Successfully")
    print("="*50)
    print("\nROWS & COLUMNS IN DATASET")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("="*50)
    
    print("\nDATATYPES OF EACH COLUMNS")
    print(df.dtypes)
    print("="*50)

    print("\n===> Checking for Missing Values <===")
    count_missing = df.isnull().sum()
    if count_missing.sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print("Missing values found in the dataset: we should handle them before proceeding with any analysis or modeling.")
    
    print("\nChecking ZEROS in dataset as in medical data if there will be zero then there are possibility of patient has not been diagnosed")
    count_zero = (df == 0).sum()
    print(count_zero)

    print("\n" + "="*50)
    print("===> Statistical Summary of Dataset <===")
    print(df.describe())
    print("="*50)

    print("\n" + "="*50)
    print("===> Correlation Analysis/Feature Analysis <===")
    target_variable = 'Outcome'
    correlation = df.corr()[target_variable].sort_values(ascending=False)
    print(correlation)
    print("="*50)
    
    for col, corr in correlation.items():
        if col == target_variable:
            continue
        if abs(corr) > 0.3:
            print(f"feature '{col}' has a strong relation with the target it can directly impact on our '{target_variable}' variable")
        elif abs(corr) > 0.1:
            print(f"feature '{col}' has a moderate relation with the target it can impact on our '{target_variable}' variable")
        else:
            print(f"feature '{col}' has a weak relation with the target it may not have a significant impact on our '{target_variable}' variable")
    
    print("="*50)
    print("\nwe might DROP features with correlation < 0.1 Why? Because they add noise, not signal, A model trained on noise will make bad predictions")

    print("\n" + "="*50)
    print("===> Distribution of Target Variable (Outcome) <===")
    counts = df[target_variable].value_counts().sort_index()
    print(counts)
    print("\nPercentage of each class in the target variable:")
    percentage = (counts / len(df) * 100)
    print(percentage)
    print("\nRatio")
    ratio = counts.min() / counts.max()
    print(f"'{ratio:.2f}:1' is the ratio of minority class to majority")
    if ratio < 0.8:
        print("The dataset is imbalanced, we should consider techniques to handle this imbalance before modeling. ALSO WE SHOULD USE F1 SCORE INSTEAD OF ACCURACY AS EVALUATION METRIC FOR IMBALANCED DATASETS TO PENALISE")
    else:
        print("The dataset is balanced, we can proceed with modeling without special handling for class imbalance.")

    print("\n" + "="*50)
    print("===> Visualization of the data <===")
    print("Let's visualize the distribution of the target variable (Outcome) using a bar plot")
    
    fig, axes = plt.subplots(figsize=(12, 5))
    counts = df[target_variable].value_counts().sort_index()
    axes.bar(counts.index, counts.values, color=['green', 'orange'])
    axes.set_title('Distribution of Target Variable (Outcome)', fontweight='bold', fontsize=14)
    axes.set_ylabel('Count')
    plt.savefig('target_distribution.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)
    ax.set_title('Feature Correlations', fontweight='bold')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    print("✓ Visualizations saved")
    
    return df