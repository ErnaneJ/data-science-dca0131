import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# xgboost ERROR
# correlation heatmap ERROR

DATA_PATH = "./diabetes_prediction_dataset.csv"

def plot_age_distribution(df):
    """
    Plot the age distribution of patients.

    @param df: DataFrame containing the dataset.
    """
    fig = px.histogram(df, x='age', nbins=30, title='Age Distribution', marginal="box", hover_data=df.columns)
    st.plotly_chart(fig)

def plot_bmi_distribution(df):
    """
    Plot the BMI distribution of patients.

    @param df: DataFrame containing the dataset.
    """
    fig = px.histogram(df, x='bmi', nbins=30, title='BMI Distribution', marginal="box", hover_data=df.columns)
    st.plotly_chart(fig)

def plot_hba1c_glucose_distribution(df):
    """
    Plot the distribution of HbA1c and blood glucose levels.

    @param df: DataFrame containing the dataset.
    """
    fig_hba1c = px.histogram(df, x='HbA1c_level', nbins=30, title='Distribution of HbA1c Levels', marginal="box", hover_data=df.columns)
    st.plotly_chart(fig_hba1c)

    fig_glucose = px.histogram(df, x='blood_glucose_level', nbins=30, title='Distribution of Blood Glucose Levels', marginal="box", hover_data=df.columns)
    st.plotly_chart(fig_glucose)

def plot_correlation_heatmap(df):
    """
    Plot a heatmap of the correlation matrix.

    @param df: DataFrame containing the dataset.
    """
    correlation_matrix = df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap')
    st.plotly_chart(fig)

def plot_comparison_by_gender(df):
    """
    Compare BMI, HbA1c levels, and blood glucose levels by gender.

    @param df: DataFrame containing the dataset.
    """
    metrics = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    labels = ['BMI', 'HbA1c Level', 'Blood Glucose Level']
    
    for metric, label in zip(metrics, labels):
        fig = px.box(df, x='gender', y=metric, title=f'{label} by Gender', hover_data=df.columns)
        st.plotly_chart(fig)

def plot_age_comparison(df):
    """
    Analyze how different age groups are associated with the presence of diabetes.

    @param df: DataFrame containing the dataset.
    """
    fig = px.histogram(df, x='age', color='diabetes', nbins=30, title='Age Distribution by Diabetes Presence', marginal="box", hover_data=df.columns)
    st.plotly_chart(fig)

def plot_outliers(df):
    """
    Identify and plot outliers in numerical variables such as BMI, HbA1c levels, and blood glucose levels.

    @param df: DataFrame containing the dataset.
    """
    metrics = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    labels = ['BMI', 'HbA1c Level', 'Blood Glucose Level']
    
    for metric, label in zip(metrics, labels):
        fig = px.box(df, y=metric, title=f'Outlier Visualization: {label}', hover_data=df.columns)
        st.plotly_chart(fig)

def plot_histograms(df):
    """
    Create histograms to visualize the distribution of each numerical variable.

    @param df: DataFrame containing the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}', marginal="box", hover_data=df.columns)
        st.plotly_chart(fig)

def plot_smoking_status(df):
    """
    Visualize the count of patients in each smoking history category.

    @param df: DataFrame containing the dataset.
    """
    fig = px.histogram(df, x='smoking_history', title='Patient Count by Smoking History', color="smoking_history", hover_data=df.columns)
    st.plotly_chart(fig)

def plot_health_conditions(df):
    """
    Use bar plots to show the distribution of hypertension and heart disease.

    @param df: DataFrame containing the dataset.
    """
    conditions = ['hypertension', 'heart_disease']
    labels = ['Hypertension', 'Heart Disease']
    
    for condition, label in zip(conditions, labels):
        fig = px.histogram(df, x=condition, title=f'Distribution of Patients with {label}', color=condition, hover_data=df.columns)
        st.plotly_chart(fig)

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
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', title='Feature Importances (RandomForest)', orientation='h')
    st.plotly_chart(fig)

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

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', title='Feature Importances (XGBoost)', orientation='h')
    st.plotly_chart(fig)

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

    fig = px.scatter(df, x='age', y='bmi', color='Cluster', title='Clusters Identified by K-means', hover_data=df.columns)
    st.plotly_chart(fig)

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
    fig = px.scatter(df, x='age', y='bmi', color='diabetes', title='Relationship between Age and BMI', hover_data=df.columns)
    st.plotly_chart(fig)

def plot_scatter_glucose_hba1c(df):
    """
    Plot the relationship between blood glucose levels and HbA1c levels.

    @param df: DataFrame containing the dataset.
    """
    fig = px.scatter(df, x='blood_glucose_level', y='HbA1c_level', color='diabetes', title='Relationship between Blood Glucose and HbA1c Levels', hover_data=df.columns)
    st.plotly_chart(fig)

def plot_pie_chart(df):
    """
    Create a pie chart showing the distribution of diabetic and non-diabetic patients.

    @param df: DataFrame containing the dataset.
    """
    diabetes_count = df['diabetes'].value_counts()
    fig = px.pie(values=diabetes_count.values, names=diabetes_count.index, title='Diabetes vs. Non-Diabetes Patients')
    st.plotly_chart(fig)

# Streamlit interface
def main():
    st.title("Diabetes Data Visualization Dashboard")

    # Load data
    df = pd.read_csv(DATA_PATH)

    st.sidebar.header("Select Visualization")
    visualization = st.sidebar.selectbox("Choose a chart to display", [
        "Age Distribution",
        "BMI Distribution",
        "HbA1c and Glucose Distribution",
        "Correlation Heatmap",
        "Comparison by Gender",
        "Age Comparison by Diabetes Presence",
        "Outliers Detection",
        "Histograms of All Numerical Variables",
        "Smoking History Distribution",
        "Health Conditions Distribution",
        "Feature Importance (RandomForest)",
        "Feature Importance (XGBoost)",
        "K-means Clustering",
        "Scatter Plot: Age vs BMI",
        "Scatter Plot: Glucose vs HbA1c",
        "Pie Chart: Diabetes Distribution"
    ])

    if visualization == "Age Distribution":
        plot_age_distribution(df)
    elif visualization == "BMI Distribution":
        plot_bmi_distribution(df)
    elif visualization == "HbA1c and Glucose Distribution":
        plot_hba1c_glucose_distribution(df)
    elif visualization == "Correlation Heatmap":
        plot_correlation_heatmap(df)
    elif visualization == "Comparison by Gender":
        plot_comparison_by_gender(df)
    elif visualization == "Age Comparison by Diabetes Presence":
        plot_age_comparison(df)
    elif visualization == "Outliers Detection":
        plot_outliers(df)
    elif visualization == "Histograms of All Numerical Variables":
        plot_histograms(df)
    elif visualization == "Smoking History Distribution":
        plot_smoking_status(df)
    elif visualization == "Health Conditions Distribution":
        plot_health_conditions(df)
    elif visualization == "Feature Importance (RandomForest)":
        plot_feature_importance(df)
    elif visualization == "Feature Importance (XGBoost)":
        plot_feature_importance_xgb(df)
    elif visualization == "K-means Clustering":
        plot_clustering(df)
    elif visualization == "Scatter Plot: Age vs BMI":
        plot_scatter_age_bmi(df)
    elif visualization == "Scatter Plot: Glucose vs HbA1c":
        plot_scatter_glucose_hba1c(df)
    elif visualization == "Pie Chart: Diabetes Distribution":
        plot_pie_chart(df)

if __name__ == "__main__":
    main()
