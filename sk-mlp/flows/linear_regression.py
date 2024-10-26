# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import mlflow
import mlflow.sklearn
import psutil
import time

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Function for data preprocessing
def preprocess_data(df):
    # Convert data types to category 
    for c in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        df[c] = df[c].astype('category')

    # Impute missing values for number columns
    for c in df.select_dtypes(exclude=['category']).columns:
        df[c] = df[c].fillna(df[c].mean())

    # Impute missing values for category columns and encode
    for c in df.select_dtypes(include=['category']).columns:
        df[c] = df[c].fillna(df[c].mode()[0])
        # Apply Label encoding to each column
        le = LabelEncoder()  
        df[c] = le.fit_transform(df[c])

    # Remove features which are not related to Target
    df = df.drop(columns=['Order', 'PID', 'Mo Sold', 'Yr Sold'])

    # Split data into X (features) and y (target)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    return X, y

# Function for training the model
def train_model(X_train, y_train):
    # Initialize the regressor
    clf = LinearRegression()

    # Train the model
    clf.fit(X_train, y_train)

    return clf

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    # 1. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print("1. Mean Absolute Error (MAE): ", mae)

    # 2. Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("2. Mean Absolute Percentage Error (MAPE): ", mape)

    # 3. Root Mean Square Error (RMSE)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("3. Root Mean Square Error (RMSE): ", rmse)

    # 4. R-squared
    r2 = r2_score(y_test, y_pred)
    print('4. R-Squared Score:', r2)

    return r2  # Return R-squared for logging

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log model metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Update metric names to be valid
        mlflow.log_metric("Mean_Absolute_Error", mae)
        mlflow.log_metric("Mean_Absolute_Percentage_Error", mape)
        mlflow.log_metric("Root_Mean_Square_Error", rmse)
        mlflow.log_metric("R_Squared_Score", r2)
        
        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        r2 = evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")

# Main function
def main():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\skrishnamurthy7\OneDrive - Schlumberger\Documents\Mtech\Semester 3\API driven cloud computing\sk-dsp\data\AmesHousing.csv")

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate and log to MLflow
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
