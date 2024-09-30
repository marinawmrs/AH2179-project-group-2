# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load the dataset
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/main/ProjectAssignmentData/Dataset-PT.csv'
df = pd.read_csv(url, skiprows=1, low_memory=False)

# Step 1: Data Preprocessing
def load_and_preprocess_data(url, skiprows, low_memory):
    df = pd.read_csv(url, skiprows=skiprows, low_memory=low_memory)
    df.ffill(inplace=True)  # Use forward fill to handle missing values
    return df

# Step 2: Feature Engineering
def feature_engineering(df):
    # Create a new column for the day of the week
    df['day_of_week'] = np.where(df['factor(day_of_week)weekday'] == 1, 'Weekday', 'Weekend')

    # Create a new column for the time of day
    df['time_of_day'] = np.where(df['factor(time_of_day)Morning_peak'] == 1, 'Morning Peak',
                                 np.where(df['factor(time_of_day)Afternoon_peak'] == 1, 'Afternoon Peak', 'Off-Peak'))

    # Create a new column for weather conditions
    weather_conditions = ['Light Rain', 'Light Snow', 'Normal', 'Rain', 'Snow']
    df['weather'] = np.select(
        [
            df['factor(weather)Light_Rain'] == 1,
            df['factor(weather)Light_Snow'] == 1,
            df['factor(weather)Normal'] == 1,
            df['factor(weather)Rain'] == 1,
            df['factor(weather)Snow'] == 1,
        ],
        weather_conditions
    )

    # Create a new column for temperature
    df['temperature'] = np.select(
        [
            df['factor(temperature)Cold'] == 1,
            df['factor(temperature)Extra_cold'] == 1,
            df['factor(temperature)Normal'] == 1,
        ],
        ['Cold', 'Extra Cold', 'Normal']
    )

    # Convert arrival_delay to numeric, if necessary
    df['arrival_delay'] = pd.to_numeric(df['arrival_delay'], errors='coerce')

    return df

# Preprocess the dataset
df = load_and_preprocess_data(url, skiprows=1, low_memory=False)
df = feature_engineering(df)

# Step 3: Exploratory Data Analysis (EDA)
# Step 3: Exploratory Data Analysis (EDA)
def perform_eda_with_outliers(df):
    # Display DataFrame information and summary statistics
    print(df.info())
    print("Summary Statistics:\n", df.describe())
    print("Missing Values:\n", df.isnull().sum())

    # Drop unwanted columns: 'column_date', 'route_id', 'bus_id', 'stop_sequence'
    df_numeric = df.drop(columns=['Calendar_date', 'route_id', 'bus_id', 'stop_sequence'], errors='ignore')

    # Select only numeric columns for correlation matrix
    numeric_df = df_numeric.select_dtypes(include='number')

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Print correlation matrix
    print("Correlation Matrix:\n", corr_matrix)

    # Find significant correlations with 'arrival_delay'
    significant_corr = corr_matrix['arrival_delay'].sort_values(ascending=False)

    # Filter variables with correlation above a threshold (e.g., 0.1 for positive and -0.1 for negative)
    threshold = 0.1
    significant_corr = significant_corr[(significant_corr > threshold) | (significant_corr < -threshold)]

    # Print significant correlations with 'arrival_delay'
    print("\nVariables with significant correlation to Arrival Delay:\n", significant_corr)

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix (Excluding Date, Route, Bus, and Stop Sequence)')
    plt.show()

    # Boxplot for Arrival Delay (overall) to visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='arrival_delay', data=df)
    plt.title('Overall Arrival Delay (Outliers Visualization)')
    plt.show()

    # Boxplot for Arrival Delay by Stop Sequence to visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='stop_sequence', y='arrival_delay', data=df)
    plt.title('Arrival Delay by Stop Sequence (Outliers Visualization)')
    plt.xticks(rotation=45)
    plt.show()

    # Boxplot for Arrival Delay by Weather Conditions to visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weather', y='arrival_delay', data=df)
    plt.title('Arrival Delay by Weather Conditions (Outliers Visualization)')
    plt.show()

    # Boxplot for Arrival Delay by Time of Day to visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='time_of_day', y='arrival_delay', data=df)
    plt.title('Arrival Delay by Time of Day (Outliers Visualization)')
    plt.show()

    # Boxplot for Arrival Delay by Temperature to visualize outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='temperature', y='arrival_delay', data=df)
    plt.title('Arrival Delay by Temperature (Outliers Visualization)')
    plt.show()

# Perform EDA with outlier visualizations
perform_eda_with_outliers(df)

# Step 5: Model Training
def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f'Trained {name}')
    return trained_models

# Step 6: Model Evaluation
def evaluate_models(trained_models, X_val, y_val):
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)

        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape
        }
        print(f"{name} - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}")

    return results

# Prepare data for modeling, including variables with significant correlation to arrival delay
X = df[['weather', 'temperature', 'day_of_week', 'time_of_day',
        'upstream_stop_delay', 'origin_delay', 'previous_bus_delay',
        'factor(day_of_week)weekend', 'factor(time_of_day)Off-peak',
        'scheduled_travel_time', 'factor(day_of_week)weekday']]

y = df['arrival_delay']

# Step 4: Model Selection
def model_selection(X, y):
    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical features
    categorical_features = ['weather', 'temperature', 'day_of_week', 'time_of_day']
    numeric_features = ['upstream_stop_delay', 'origin_delay', 'previous_bus_delay',
                        'factor(day_of_week)weekend', 'factor(time_of_day)Off-peak',
                        'scheduled_travel_time', 'factor(day_of_week)weekday']

    # Preprocessing pipeline for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    # Define models
    models = {
        'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor())]),
        'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', LinearRegression())]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', XGBRegressor())])
    }

    return X_train, X_val, y_train, y_val, models

# Model selection
X_train, X_val, y_train, y_val, models = model_selection(X, y)

# Train models
trained_models = train_models(models, X_train, y_train)

# Evaluate models
evaluation_results = evaluate_models(trained_models, X_val, y_val)

# Display final evaluation results
print("Evaluation Results:\n", evaluation_results)
