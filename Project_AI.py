import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset and skip the first row and use the second row as headers
data = pd.read_csv('Dataset-PT.csv', skiprows=1, low_memory=False)

# Data Exploration
with open('model_results.txt', 'w') as file:
    file.write("Data Columns: \n")
    file.write(str(data.columns) + "\n\n")
    file.write("Data Info: \n")
    file.write(str(data.info()) + "\n\n")
    file.write("Summary Statistics: \n")
    file.write(str(data.describe()) + "\n\n")

    # Check for missing values
    file.write("Missing Values: \n")
    file.write(str(data.isnull().sum()) + "\n\n")

# Create a dictionary to store the results
results = {}

# 1. Overall Average Arrival Delay
overall_avg_delay = data['arrival_delay'].mean()
results['overall_avg_delay'] = [overall_avg_delay]  # Adding as a list to match DataFrame format
print(f"Overall Average Delay: {overall_avg_delay}")

# 2. Average Arrival Delay by Stop Sequence
avg_delay_per_stop = data.groupby('stop_sequence')['arrival_delay'].mean().reset_index()
avg_delay_per_stop.columns = ['stop_sequence', 'avg_delay_stop']

# 3. Average Arrival Delay by Day of the Week
avg_delay_per_day = data.groupby('day_of_week')['arrival_delay'].mean().reset_index()
avg_delay_per_day.columns = ['day_of_week', 'avg_delay_day']

# 4. Average Arrival Delay by Time of Day
avg_delay_per_time = data.groupby('time_of_day')['arrival_delay'].mean().reset_index()
avg_delay_per_time.columns = ['time_of_day', 'avg_delay_time']

# 5. Average Arrival Delay by Weather Condition
avg_delay_per_weather = data.groupby('weather')['arrival_delay'].mean().reset_index()
avg_delay_per_weather.columns = ['weather', 'avg_delay_weather']

# 6. Average Arrival Delay by Temperature
avg_delay_per_temperature = data.groupby('temperature')['arrival_delay'].mean().reset_index()
avg_delay_per_temperature.columns = ['temperature', 'avg_delay_temperature']

# Combine all the results into a single DataFrame
merged_results_df = pd.concat([pd.DataFrame(results), avg_delay_per_stop, avg_delay_per_day,
                               avg_delay_per_time, avg_delay_per_weather, avg_delay_per_temperature], axis=1)

# Save the combined results to a single CSV file
merged_results_df.to_csv('combined_arrival_delay_results_with_stop.csv', index=False)

# Output the merged DataFrame to verify
print(merged_results_df)

# --- Step 2: Conclusion of Variables ---

# Initialize dictionary to store conclusions
conclusions = {}

# Frequency and maximum analysis for each categorical variable
# 1. Conclusion for Time of Day
time_above_avg = data[data['arrival_delay'] > overall_avg_delay].groupby('time_of_day').size().reset_index(name='count')
time_highest_frequency = time_above_avg.loc[time_above_avg['count'].idxmax()]
highest_avg_time = avg_delay_per_time.loc[avg_delay_per_time['avg_delay_time'].idxmax()]

conclusions['time_of_day'] = {
    'most_frequent_above_avg': time_highest_frequency['time_of_day'],
    'highest_avg_delay': highest_avg_time['time_of_day']
}

print(f"Conclusion for Time of Day:\n"
      f"Time of Day with the most delays above the average: {time_highest_frequency['time_of_day']}\n"
      f"Time of Day with the highest average delay: {highest_avg_time['time_of_day']}\n")

# 2. Conclusion for Weather
weather_above_avg = data[data['arrival_delay'] > overall_avg_delay].groupby('weather').size().reset_index(name='count')
weather_highest_frequency = weather_above_avg.loc[weather_above_avg['count'].idxmax()]
highest_avg_weather = avg_delay_per_weather.loc[avg_delay_per_weather['avg_delay_weather'].idxmax()]

conclusions['weather'] = {
    'most_frequent_above_avg': weather_highest_frequency['weather'],
    'highest_avg_delay': highest_avg_weather['weather']
}

print(f"Conclusion for Weather:\n"
      f"Weather condition with the most delays above the average: {weather_highest_frequency['weather']}\n"
      f"Weather condition with the highest average delay: {highest_avg_weather['weather']}\n")

# 3. Conclusion for Temperature
temperature_above_avg = data[data['arrival_delay'] > overall_avg_delay].groupby('temperature').size().reset_index(name='count')
temperature_highest_frequency = temperature_above_avg.loc[temperature_above_avg['count'].idxmax()]
highest_avg_temperature = avg_delay_per_temperature.loc[avg_delay_per_temperature['avg_delay_temperature'].idxmax()]

conclusions['temperature'] = {
    'most_frequent_above_avg': temperature_highest_frequency['temperature'],
    'highest_avg_delay': highest_avg_temperature['temperature']
}

print(f"Conclusion for Temperature:\n"
      f"Temperature condition with the most delays above the average: {temperature_highest_frequency['temperature']}\n"
      f"Temperature condition with the highest average delay: {highest_avg_temperature['temperature']}\n")

# 4. Conclusion for Day of the Week
day_above_avg = data[data['arrival_delay'] > overall_avg_delay].groupby('day_of_week').size().reset_index(name='count')
day_highest_frequency = day_above_avg.loc[day_above_avg['count'].idxmax()]
highest_avg_day = avg_delay_per_day.loc[avg_delay_per_day['avg_delay_day'].idxmax()]

conclusions['day_of_week'] = {
    'most_frequent_above_avg': day_highest_frequency['day_of_week'],
    'highest_avg_delay': highest_avg_day['day_of_week']
}

print(f"Conclusion for Day of the Week:\n"
      f"Day of the Week with the most delays above the average: {day_highest_frequency['day_of_week']}\n"
      f"Day of the Week with the highest average delay: {highest_avg_day['day_of_week']}\n")

# 5. Conclusion for Stop Sequence
stop_above_avg = data[data['arrival_delay'] > overall_avg_delay].groupby('stop_sequence').size().reset_index(name='count')
stop_highest_frequency = stop_above_avg.loc[stop_above_avg['count'].idxmax()]
highest_avg_stop = avg_delay_per_stop.loc[avg_delay_per_stop['avg_delay_stop'].idxmax()]

conclusions['stop_sequence'] = {
    'most_frequent_above_avg': stop_highest_frequency['stop_sequence'],
    'highest_avg_delay': highest_avg_stop['stop_sequence']
}

print(f"Conclusion for Stop Sequence:\n"
      f"Stop with the most delays above the average: {stop_highest_frequency['stop_sequence']}\n"
      f"Stop with the highest average delay: {highest_avg_stop['stop_sequence']}\n")

# Save conclusions to a CSV file
conclusions_df = pd.DataFrame.from_dict(conclusions, orient='index').reset_index()
conclusions_df.columns = ['variable', 'most_frequent_above_avg', 'highest_avg_delay']
conclusions_df.to_csv('conclusions_arrival_delay_with_stop.csv', index=False)

# Output the conclusions DataFrame to verify
print(conclusions_df)

# Create new columns for day of the week, time of day, weather, and temperature
data['day_of_week'] = np.where(data['factor(day_of_week)weekday'] == 1, 'Weekday', 'Weekend')
data['time_of_day'] = np.where(data['factor(time_of_day)Morning_peak'] == 1, 'Morning Peak',
                               np.where(data['factor(time_of_day)Afternoon_peak'] == 1, 'Afternoon Peak',
                                        'Off-Peak'))

# Create a new column for weather conditions
weather_conditions = ['Light Rain', 'Light Snow', 'Normal', 'Rain', 'Snow']
data['weather'] = np.select(
    [
        data['factor(weather)Light_Rain'] == 1,
        data['factor(weather)Light_Snow'] == 1,
        data['factor(weather)Normal'] == 1,
        data['factor(weather)Rain'] == 1,
        data['factor(weather)Snow'] == 1,
    ],
    weather_conditions
)

# Create a new column for temperature
data['temperature'] = np.select(
    [
        data['factor(temperature)Cold'] == 1,
        data['factor(temperature)Extra_cold'] == 1,
        data['factor(temperature)Normal'] == 1,
    ],
    ['Cold', 'Extra Cold', 'Normal']
)

# Histogram for the distribution of arrival delays
# Plotting the overall arrival delay distribution
plt.figure(figsize=(14, 7))

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(14, 7))

# Histogram for the distribution of arrival delays with customized aesthetics
sns.histplot(data['arrival_delay'], bins=30, kde=True, color='green', edgecolor='black', alpha=1.0)

# Add titles and labels
plt.title('Overall Arrival Delay Distribution', fontsize=20, fontweight='bold')
plt.xlabel('Arrival Delay (seconds)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# Average Arrival Delay by Stop Sequence
avg_delay_per_stop = data.groupby(['stop_sequence'])['arrival_delay'].mean().reset_index()
print("Average Arrival Delay by Stop Sequence:")
print(avg_delay_per_stop)

# Save to CSV
avg_delay_per_stop.to_csv('average_arrival_delay_by_stop_sequence.csv', index=False)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the bar plot
plt.figure(figsize=(14, 7))

# Barplot with customized aesthetics
bar_plot = sns.barplot(data=avg_delay_per_stop, x='stop_sequence', y='arrival_delay')

# Add titles and labels
plt.title('Average Arrival Delay by Stop Sequence', fontsize=20, fontweight='bold')
plt.xlabel('Stop Sequence', fontsize=14)
plt.ylabel('Average Arrival Delay (seconds)', fontsize=14)

# Customize ticks
plt.xticks(rotation=45, fontsize=12)  # Rotate x labels for better visibility
plt.yticks(fontsize=12)

# Add data labels on top of the bars
for p in bar_plot.patches:
    bar_plot.annotate(f'{p.get_height():.1f}',
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', fontsize=10, color='black')

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# 1. Average Arrival Delay by Day of the Week
avg_delay_per_day = data.groupby(['stop_sequence', 'day_of_week'])['arrival_delay'].mean().reset_index()
print("Average Arrival Delay by Day of the Week:")
print(avg_delay_per_day)

# Save to CSV
avg_delay_per_day.to_csv('average_arrival_delay_by_day_of_week.csv', index=False)

plt.figure(figsize=(14, 7))
sns.barplot(data=avg_delay_per_day, x='stop_sequence', y='arrival_delay', hue='day_of_week', palette='viridis')
plt.title('Average Arrival Delay by Stop Sequence (Weekday vs Weekend)')
plt.xlabel('Stop Sequence')
plt.ylabel('Average Arrival Delay (seconds)')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.legend(title='Day Type')
plt.show()

# 2. Average Arrival Delay by Time of Day
avg_delay_per_time = data.groupby(['stop_sequence', 'time_of_day'])['arrival_delay'].mean().reset_index()
print("Average Arrival Delay by Time of Day:")
print(avg_delay_per_time)

# Save to CSV
avg_delay_per_time.to_csv('average_arrival_delay_by_time_of_day.csv', index=False)

plt.figure(figsize=(14, 7))
sns.barplot(data=avg_delay_per_time, x='stop_sequence', y='arrival_delay', hue='time_of_day', palette='coolwarm')
plt.title('Average Arrival Delay by Stop Sequence (Time of Day)')
plt.xlabel('Stop Sequence')
plt.ylabel('Average Arrival Delay (seconds)')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.legend(title='Time of Day')
plt.show()

# 3. Average Arrival Delay by Weather Condition
avg_delay_per_weather = data.groupby(['stop_sequence', 'weather'])['arrival_delay'].mean().reset_index()
print("Average Arrival Delay by Weather Condition:")
print(avg_delay_per_weather)

# Save to CSV
avg_delay_per_weather.to_csv('average_arrival_delay_by_weather_condition.csv', index=False)

plt.figure(figsize=(14, 7))
sns.barplot(data=avg_delay_per_weather, x='stop_sequence', y='arrival_delay', hue='weather', palette='viridis')
plt.title('Average Arrival Delay by Stop Sequence (Weather Conditions)')
plt.xlabel('Stop Sequence')
plt.ylabel('Average Arrival Delay (seconds)')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.legend(title='Weather Condition')
plt.show()

# 4. Average Arrival Delay by Temperature
avg_delay_per_temperature = data.groupby(['stop_sequence', 'temperature'])['arrival_delay'].mean().reset_index()
print("Average Arrival Delay by Temperature:")
print(avg_delay_per_temperature)

# Save to CSV
avg_delay_per_temperature.to_csv('average_arrival_delay_by_temperature.csv', index=False)

plt.figure(figsize=(14, 7))
sns.barplot(data=avg_delay_per_temperature, x='stop_sequence', y='arrival_delay', hue='temperature', palette='coolwarm')
plt.title('Average Arrival Delay by Stop Sequence (Temperature)')
plt.xlabel('Stop Sequence')
plt.ylabel('Average Arrival Delay (seconds)')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.legend(title='Temperature')
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Target and features
X = data.drop(columns=['arrival_delay'])
y = data['arrival_delay']

# Convert delay into categories for classification
y_class = pd.cut(y, bins=[-np.inf, 2, 10, np.inf], labels=['On-time', 'Moderate Delay', 'Severe Delay'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Drop the original categorical columns (weather, temperature, day_of_week, time_of_day)
X_train = X_train.drop(columns=['weather', 'temperature', 'day_of_week', 'time_of_day'])
X_test = X_test.drop(columns=['weather', 'temperature', 'day_of_week', 'time_of_day'])
X_train_class = X_train_class.drop(columns=['weather', 'temperature', 'day_of_week', 'time_of_day'])
X_test_class = X_test_class.drop(columns=['weather', 'temperature', 'day_of_week', 'time_of_day'])

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Initialize variables to track the best model and its parameters
best_model = None
best_mae = float('inf')
best_accuracy = 0
best_params = {}

# File to store results
with open('model_results.txt', 'a') as file:
    # 1. Regression: Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    file.write(f"Linear Regression MAE: {mae_lr}, MSE: {mse_lr}, R²: {r2_lr}\n\n")

    # Plot actual vs predicted for Linear Regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', lw=2)
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual Arrival Delay')
    plt.ylabel('Predicted Arrival Delay')
    plt.savefig('linear_regression_actual_vs_predicted.png')
    plt.close()

    # 2. Regression: Random Forest Regressor with Hyperparameter Tuning
    rf_reg = RandomForestRegressor(random_state=42)
    param_grid_rf_reg = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf_reg = GridSearchCV(rf_reg, param_grid_rf_reg, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
    grid_rf_reg.fit(X_train_scaled, y_train)
    y_pred_rf_reg = grid_rf_reg.best_estimator_.predict(X_test_scaled)
    mae_rf_reg = mean_absolute_error(y_test, y_pred_rf_reg)
    mse_rf_reg = mean_squared_error(y_test, y_pred_rf_reg)
    r2_rf_reg = r2_score(y_test, y_pred_rf_reg)
    file.write(f"Random Forest Regressor (tuned) MAE: {mae_rf_reg}, MSE: {mse_rf_reg}, R²: {r2_rf_reg}\n")
    file.write(f"Best Random Forest Regressor Parameters: {grid_rf_reg.best_params_}\n\n")

    # Plot feature importance for Random Forest Regressor
    feature_importance = grid_rf_reg.best_estimator_.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(range(X_train.shape[1]), feature_importance[sorted_idx], align='center')
    plt.yticks(range(X_train.shape[1]), X.columns[sorted_idx])
    plt.title('Random Forest Regressor Feature Importance')
    plt.savefig('random_forest_regressor_feature_importance.png')
    plt.close()

    # 3. Classification: Logistic Regression with Class Weights for Imbalance
    log_reg = LogisticRegression(max_iter=500, class_weight='balanced')
    log_reg.fit(X_train_class_scaled, y_train_class)
    y_pred_log_reg = log_reg.predict(X_test_class_scaled)
    accuracy_log_reg = accuracy_score(y_test_class, y_pred_log_reg)
    file.write(f"Logistic Regression Accuracy: {accuracy_log_reg}\n")
    file.write(str(classification_report(y_test_class, y_pred_log_reg)) + "\n\n")

    # Plot confusion matrix for Logistic Regression
    confusion_matrix_log_reg = confusion_matrix(y_test_class, y_pred_log_reg)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('logistic_regression_confusion_matrix.png')
    plt.close()

    # 4. Classification: Random Forest Classifier with Hyperparameter Tuning
    rf_class = RandomForestClassifier(random_state=42)
    param_grid_rf_class = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf_class = GridSearchCV(rf_class, param_grid_rf_class, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf_class.fit(X_train_class_scaled, y_train_class)
    y_pred_rf_class = grid_rf_class.best_estimator_.predict(X_test_class_scaled)
    accuracy_rf_class = accuracy_score(y_test_class, y_pred_rf_class)
    file.write(f"Random Forest Classifier (tuned) Accuracy: {accuracy_rf_class}\n")
    file.write(str(classification_report(y_test_class, y_pred_rf_class)) + "\n")
    file.write(f"Best Random Forest Classifier Parameters: {grid_rf_class.best_params_}\n\n")

    # Plot confusion matrix for Random Forest Classifier
    confusion_matrix_rf_class = confusion_matrix(y_test_class, y_pred_rf_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_rf_class, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Random Forest Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('random_forest_classifier_confusion_matrix.png')
    plt.close()

# Save the best models
joblib.dump(grid_rf_reg.best_estimator_, 'best_rf_regressor.pkl')
joblib.dump(grid_rf_class.best_estimator_, 'best_rf_classifier.pkl')

# Summarize the best model
with open('model_results.txt', 'a') as file:
    file.write(f"Best Model: {best_model}\n")
    if best_model in ["Random Forest Regressor", "Random Forest Classifier"]:
        file.write(f"Best Model Parameters: {best_params}\n")
