# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
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

# Function for training the model with hyperparameter tuning
def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
        }
    
    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    print("1. Mean Absolute Error (MAE): ", mae)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("2. Mean Absolute Percentage Error (MAPE): ", mape)

    rmse = root_mean_squared_error(y_test, y_pred)
    print("3. Root Mean Square Error (RMSE): ", rmse)

    r2 = r2_score(y_test, y_pred)
    print('4. R-Squared Score (R2):', r2)

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log hyper parameters
        if isinstance(model, RandomForestRegressor):
            mlflow.log_param("model_type", "Random Forest")
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        elif isinstance(model, GradientBoostingRegressor):
            mlflow.log_param("model_type", "Gradient Boosting")
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        
        # Log model metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Absolute Percentage Error", mape)
        mlflow.log_metric("Root Mean Square Error", rmse)
        mlflow.log_metric("R-Squared Score", r2)
        
        # Log system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        mlflow.log_metric("execution_time", end_time - start_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

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

    # Train the model (using Random Forest for example)
    model = train_model(X_train, y_train, model_type='rf')  # Change to 'gb' for Gradient Boosting

    # Evaluate and log to MLflow
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
