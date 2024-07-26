import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

plt.style.use('classic')
sns.set()

DATA_PATH = "../../data/diabetes_prediction_dataset.csv"
CHARTS_DIR = "./charts/current/"

def save_and_show_plot(title):
    """
    Save the plot to a file and display it.

    @param title: The title of the plot to be used as the filename.
    """
    if not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(CHARTS_DIR, f"{timestamp}_{title}.png")
    plt.savefig(filename)
    plt.show()

def plot_age_distribution(df):
    """
    Plot the age distribution of patients.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="Age Distribution", figsize=(10, 5))
    sns.histplot(df['age'], kde=True, bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    save_and_show_plot('age_distribution')

def plot_bmi_distribution(df):
    """
    Plot the BMI distribution of patients.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="BMI Distribution", figsize=(10, 5))
    sns.histplot(df['bmi'], kde=True, bins=30)
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    save_and_show_plot('bmi_distribution')

def plot_hba1c_glucose_distribution(df):
    """
    Plot the distribution of HbA1c and blood glucose levels.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="Distribution of HbA1c and Blood Glucose Levels", figsize=(10, 5))
    sns.histplot(df['HbA1c_level'], kde=True, bins=30, color='blue', label='HbA1c Level')
    plt.title('Distribution of HbA1c and Blood Glucose Levels')
    plt.xlabel('Level')
    plt.ylabel('Frequency')
    plt.legend()
    save_and_show_plot('hba1c_distribution')

    plt.figure(num="Blood Glucose Level", figsize=(10, 5))
    sns.histplot(df['blood_glucose_level'], kde=True, bins=30, color='red', label='Blood Glucose Level')
    plt.xlabel('Blood Glucose Level')
    plt.ylabel('Frequency')
    plt.legend()
    save_and_show_plot('glucose_distribution')

def plot_correlation_matrix(df):
    """
    Calculate and plot the correlation matrix of numerical variables.

    @param df: DataFrame containing the dataset.
    @return: Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a heatmap of the correlation matrix.

    @param correlation_matrix: DataFrame containing the correlation matrix.
    """
    plt.figure(num="Correlation Heatmap", figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    save_and_show_plot('correlation_heatmap')

def plot_comparison_by_gender(df):
    """
    Compare BMI, HbA1c levels, and blood glucose levels by gender.

    @param df: DataFrame containing the dataset.
    """
    metrics = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    labels = ['BMI', 'HbA1c Level', 'Blood Glucose Level']
    
    for metric, label in zip(metrics, labels):
        plt.figure(num="", figsize=(10, 5))
        sns.boxplot(x='gender', y=metric, data=df)
        plt.title(f'{label} by Gender')
        plt.xlabel('Gender')
        plt.ylabel(label)
        save_and_show_plot(f'comparison_by_gender_{metric}')

def plot_age_comparison(df):
    """
    Analyze how different age groups are associated with the presence of diabetes.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="Age Distribution by Diabetes Presence", figsize=(10, 5))
    sns.histplot(data=df, x='age', hue='diabetes', multiple='stack', bins=30, kde=True)
    plt.title('Age Distribution by Diabetes Presence')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(title='Diabetes', labels=['No Diabetes', 'Diabetes'])
    save_and_show_plot('age_comparison')

def plot_outliers(df):
    """
    Identify and plot outliers in numerical variables such as BMI, HbA1c levels, and blood glucose levels.

    @param df: DataFrame containing the dataset.
    """
    metrics = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    labels = ['BMI', 'HbA1c Level', 'Blood Glucose Level']
    
    for metric, label in zip(metrics, labels):
        plt.figure(num="", figsize=(10, 5))
        sns.boxplot(data=df, y=metric)
        plt.title(f'Outlier Visualization: {label}')
        plt.xlabel(label)
        save_and_show_plot(f'outliers_{metric}')
        
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
        print(f"\nOutliers for {label}:")
        print(outliers[[metric]].describe())

def plot_histograms(df):
    """
    Create histograms to visualize the distribution of each numerical variable.

    @param df: DataFrame containing the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        plt.figure(num=f'Distribution of {col}', figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        save_and_show_plot(f'histogram_{col}')

def plot_boxplots(df):
    """
    Use boxplots to summarize the distribution of numerical variables and detect outliers.

    @param df: DataFrame containing the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        plt.figure(num=f'Boxplot of {col}', figsize=(10, 5))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        save_and_show_plot(f'boxplot_{col}')

def plot_smoking_status(df):
    """
    Visualize the count of patients in each smoking history category.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num='Patient Count by Smoking History', figsize=(10, 5))
    sns.countplot(x='smoking_history', data=df)
    plt.title('Patient Count by Smoking History')
    plt.xlabel('Smoking History')
    plt.ylabel('Count')
    save_and_show_plot('smoking_status')

def plot_health_conditions(df):
    """
    Use bar plots to show the distribution of hypertension and heart disease.

    @param df: DataFrame containing the dataset.
    """
    conditions = ['hypertension', 'heart_disease']
    labels = ['Hypertension', 'Heart Disease']
    
    for condition, label in zip(conditions, labels):
        plt.figure(num=f'Distribution of Patients with {label}', figsize=(10, 5))
        sns.countplot(x=condition, data=df)
        plt.title(f'Distribution of Patients with {label}')
        plt.xlabel(label)
        plt.ylabel('Count')
        save_and_show_plot(f'health_conditions_{condition}')

def plot_feature_importance(df):
    """
    Evaluate the importance of features in predicting diabetes using RandomForestClassifier.

    @param df: DataFrame containing the dataset.
    """
    df, label_encoders = encode_categorical_features(df)
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_

    plt.figure(num='Feature Importances (RandomForest)', figsize=(10, 5))
    features = X.columns
    sorted_indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), features[sorted_indices], rotation=90)
    plt.title('Feature Importances (RandomForest)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    save_and_show_plot('feature_importance_rf')

def plot_feature_importance_xgb(df):
    """
    Evaluate the importance of features in predicting diabetes using XGBoost.

    @param df: DataFrame containing the dataset.
    """
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(num="Feature Importances (XGBoost)", figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances (XGBoost)')
    save_and_show_plot('feature_importance_xgb')

def plot_clustering(df):
    """
    Apply K-means clustering to identify hidden patterns in the data.

    @param df: DataFrame containing the dataset.
    """
    df, label_encoders = encode_categorical_features(df)

    features = df.drop('diabetes', axis=1)

    kmeans = KMeans(n_clusters=9, random_state=42)
    clusters = kmeans.fit_predict(features)

    df['Cluster'] = clusters

    plt.figure(num="Clusters Identified by K-means", figsize=(10, 5))
    sns.scatterplot(data=df, x='age', y='bmi', hue='Cluster', palette='viridis', alpha=0.7)
    plt.title('Clusters Identified by K-means')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.legend(title='Cluster')
    save_and_show_plot('clustering')

def encode_categorical_features(df):
    """
    Encode categorical variables using LabelEncoder.

    @param df: DataFrame containing the dataset.
    @return: Encoded DataFrame and dictionary of label encoders.
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def plot_scatter_age_bmi(df):
    """
    Plot the relationship between age and BMI.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="Relationship between Age and BMI", figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='bmi', hue='diabetes', palette='viridis', alpha=0.7)
    plt.title('Relationship between Age and BMI')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.legend(title='Diabetes', labels=['No Diabetes', 'Diabetes'])
    save_and_show_plot('scatter_age_bmi')

def plot_scatter_glucose_hba1c(df):
    """
    Plot the relationship between blood glucose levels and HbA1c levels.

    @param df: DataFrame containing the dataset.
    """
    plt.figure(num="", figsize=(10, 6))
    sns.scatterplot(data=df, x='blood_glucose_level', y='HbA1c_level', hue='diabetes', palette='viridis', alpha=0.7)
    plt.title('Relationship between Blood Glucose and HbA1c Levels')
    plt.xlabel('Blood Glucose Level')
    plt.ylabel('HbA1c Level')
    plt.legend(title='Diabetes', labels=['No Diabetes', 'Diabetes'])
    save_and_show_plot('scatter_glucose_hba1c')

def plot_pairplots(df):
    """
    Plot pairplots to visualize the relationships between multiple variables.

    @param df: DataFrame containing the dataset.
    """
    numeric_columns = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
    pairplot_df = df[numeric_columns + ['diabetes']]
    
    pairplot = sns.pairplot(pairplot_df, hue='diabetes', palette='viridis')
    pairplot.fig.suptitle('Pairplots of Variables', y=1.02)
    save_and_show_plot('pairplots')

def analyze_hypertension_diabetes(df):
    """
    Analyze and visualize patients with both hypertension and diabetes.

    @param df: DataFrame containing the dataset.
    """
    subset_df = df[(df['hypertension'] == 1) & (df['diabetes'] == 1)]
    
    print("Descriptive Statistics for Patients with Hypertension and Diabetes:")
    print(subset_df.describe())
    
    plt.figure(num="Age Distribution of Patients with Hypertension and Diabetes", figsize=(12, 6))
    sns.histplot(subset_df['age'], kde=True, bins=30)
    plt.title('Age Distribution of Patients with Hypertension and Diabetes')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    save_and_show_plot('hypertension_diabetes_age_distribution')

    plt.figure(num="BMI Distribution of Patients with Hypertension and Diabetes", figsize=(12, 6))
    sns.histplot(subset_df['bmi'], kde=True, bins=30)
    plt.title('BMI Distribution of Patients with Hypertension and Diabetes')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    save_and_show_plot('hypertension_diabetes_bmi_distribution')

def compare_smoking_status(df):
    """
    Compare metrics between patients with different smoking histories.

    @param df: DataFrame containing the dataset.
    """
    smoking_statuses = df['smoking_history'].unique()
    print(f"Smoking Histories Found: {smoking_statuses}")
    
    plt.figure(num="BMI by Smoking History", figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='smoking_history', y='bmi')
    plt.title('BMI by Smoking History')
    plt.xlabel('Smoking History')
    plt.ylabel('BMI')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='smoking_history', y='blood_glucose_level')
    plt.title('Blood Glucose Level by Smoking History')
    plt.xlabel('Smoking History')
    plt.ylabel('Blood Glucose Level')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='smoking_history', y='HbA1c_level')
    plt.title('HbA1c Level by Smoking History')
    plt.xlabel('Smoking History')
    plt.ylabel('HbA1c Level')

    plt.tight_layout()
    save_and_show_plot('smoking_status_comparison')

def main():
    """
    Main function to execute the analysis pipeline.
    """
    df = pd.read_csv(DATA_PATH)

    plot_age_distribution(df)
    plot_bmi_distribution(df)
    plot_hba1c_glucose_distribution(df)

    correlation_matrix = plot_correlation_matrix(df)
    plot_correlation_heatmap(correlation_matrix)

    plot_comparison_by_gender(df)
    plot_age_comparison(df)

    plot_outliers(df)

    plot_histograms(df)
    plot_boxplots(df)

    plot_smoking_status(df)
    plot_health_conditions(df)

    plot_feature_importance(df)
    plot_feature_importance_xgb(df)
    plot_clustering(df)

    plot_scatter_age_bmi(df)
    plot_scatter_glucose_hba1c(df)
    plot_pairplots(df)

    analyze_hypertension_diabetes(df)
    compare_smoking_status(df)

if __name__ == "__main__":
    main()
