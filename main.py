import os
import pandas as pd

def show_csv_info(file_path):
    """
    Display information about a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame containing the CSV data.
    """
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size / (1024*1024):.2f} MB")

    data = pd.read_csv(file_path)

    num_rows, num_cols = data.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")

    print("\nAttributes and data types:")
    print(data.dtypes)

    print("\nDataFrame description:")
    print(data.describe())

    print("\nQuantidade de Dados Faltantes:")
    print((data == "").sum())
    # print(data.isnull().sum())

    print("\nAdditional information:")
    print("""
    The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative).
    The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.
    This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information.
    This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans.
    Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.
    """)

    return data

file_path = 'data/diabetes_prediction_dataset.csv'
show_csv_info(file_path)
