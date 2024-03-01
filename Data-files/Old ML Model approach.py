import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Loading the CSV files for analysis
file_paths = {
    "D19": "D19.csv",
    "D21": "D21.csv",
    "GP21": "GP21.csv",
    "MSE21": "MSE21.csv",
    "NP21": "NP21.csv",
    "SRM22": "SRM22.csv",
    "SRM23": "SRM23.csv"
}

# Reading the CSV files into dataframes
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Displaying the first few rows of each dataframe to understand their structure
for name, df in dataframes.items():
    print(df.columns)
    print(f"First few rows of {name} dataset:")
    print(df.head(), "\n")


def process_data_adjusted(df):
    """
    Adjusted function to process the dataframe to extract and format the registration date and count of registrations.
    This function includes error handling for date formatting issues.
    """
    # Dropping the first row which contains column descriptions and not actual data
    df = df.iloc[1:]

    # Renaming columns for clarity
    df.columns = ['BookingReference', 'CreatedDate', 'Reference', 'AttendeeStatus', 'Attended']

    # Trying different date formats if default format fails
    for date_format in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']:
        try:
            df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], format=date_format)
            break
        except (ValueError, TypeError):
            continue

    # Filtering only 'Attending' status
    df = df[df['AttendeeStatus'] == 'Attending']

    # Grouping by date and counting registrations
    df_grouped = df.groupby('CreatedDate').size().reset_index(name='Count')

    return df_grouped

# Re-applying the adjusted processing function to each dataframe
processed_data_adjusted = {name: process_data_adjusted(df.copy()) for name, df in dataframes.items()}

# Plotting the registration trends for each conference
plt.figure(figsize=(15, 10))

for name, df in processed_data_adjusted.items():
    if not df.empty:
        plt.plot(df['CreatedDate'], df['Count'].cumsum(), label=name)

plt.xlabel('Date')
plt.ylabel('Cumulative Number of Registrations')
plt.title('Cumulative Registration Trends Over Time for Each Conference')
plt.legend()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

# Performing a statistical analysis to identify key patterns and trends
# We'll use one of the datasets as an example for this analysis
srm22_data = processed_data_adjusted['SRM22'].copy()
print(srm22_data)

# Making sure the data is sorted by date
srm22_data.sort_values('CreatedDate', inplace=True)

# Setting the date as the index for time series analysis
srm22_data.set_index('CreatedDate', inplace=True)

# Resampling the data by day to fill missing dates with zeros (no registration on those days)
srm22_data_resampled = srm22_data.resample('D').asfreq(fill_value=0)

# Decomposing the time series to observe trends, seasonality, and residuals
decomposition_srm22 = seasonal_decompose(srm22_data_resampled['Count'], model='additive')

# Plotting the decomposed time series components
plt.figure(figsize=(14, 8))

plt.subplot(411)
plt.plot(decomposition_srm22.observed, label='Observed')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(decomposition_srm22.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition_srm22.seasonal, label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition_srm22.resid, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# For demonstration, we'll use the SRM22 data for model training and validation
# Preparing the data for time series forecasting
y_srm22 = srm22_data_resampled['Count'].values
print(y_srm22)

# Since ARIMA and linear regression require different data formats, we'll prepare both
# For ARIMA, we use the entire series, but for linear regression, we need to create a feature matrix
X_srm22 = np.arange(len(y_srm22)).reshape(-1, 1)
print(X_srm22)

# Splitting the data into training and testing sets for linear regression model
X_train_srm22, X_test_srm22, y_train_srm22, y_test_srm22 = train_test_split(X_srm22, y_srm22, test_size=0.2, random_state=42)


# Function to train and evaluate machine learning models
def train_evaluate_ml_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the given machine learning model.
    Returns the trained model and the mean squared error.
    """
    # Training the model
    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, mse

# Preparing the data for machine learning models
X_srm22_ml = np.arange(len(y_srm22)).reshape(-1, 1)
X_train_srm22_ml, X_test_srm22_ml, y_train_srm22_ml, y_test_srm22_ml = train_test_split(X_srm22_ml, y_srm22, test_size=0.2, random_state=42)

# Training and evaluating Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_trained_model, rf_mse = train_evaluate_ml_model(rf_model, X_train_srm22_ml, X_test_srm22_ml, y_train_srm22_ml, y_test_srm22_ml)

# Training and evaluating Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_trained_model, gb_mse = train_evaluate_ml_model(gb_model, X_train_srm22_ml, X_test_srm22_ml, y_train_srm22_ml, y_test_srm22_ml)

print(rf_mse, gb_mse)

# Function to optimize a machine learning model using GridSearchCV
def optimize_ml_model(model, param_grid, X_train, y_train):
    """
    Optimize the given machine learning model using GridSearchCV.
    Returns the best model and its parameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

# Parameter grid for Random Forest and Gradient Boosting models
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Optimizing Random Forest model
rf_optimized_model, rf_best_params = optimize_ml_model(RandomForestRegressor(random_state=42), rf_param_grid, X_train_srm22_ml, y_train_srm22_ml)

# Optimizing Gradient Boosting model
gb_optimized_model, gb_best_params = optimize_ml_model(GradientBoostingRegressor(random_state=42), gb_param_grid, X_train_srm22_ml, y_train_srm22_ml)

# Output the best parameters
print("Best Parameters for Random Forest:", rf_best_params)
print("Best Parameters for Gradient Boosting:", gb_best_params)

# Evaluate the best models on the test set
rf_predictions = rf_optimized_model.predict(X_test_srm22)
rf_mse = mean_squared_error(y_test_srm22, rf_predictions)
print("Random Forest MSE:", rf_mse)

gb_predictions = gb_optimized_model.predict(X_test_srm22)
gb_mse = mean_squared_error(y_test_srm22, gb_predictions)
print("Gradient Boosting MSE:", gb_mse)

# Performing Cross-Validation
def cross_validate_models(datasets, model):
    results = {}
    for i, (X_train, y_train) in enumerate(datasets):
        model.fit(X_train, y_train)
        for j, (X_test, y_test) in enumerate(datasets):
            if i != j:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                results[f'Train on {i}, Test on {j}'] = mse
    return results

# Prepare the datasets for cross-validation
datasets_for_cv = [process_data_adjusted(dataframes[name].copy()) for name in dataframes]

# Convert each dataset for ML model input
datasets_for_ml = [(np.arange(len(df)).reshape(-1, 1), df['Count'].values) for df in datasets_for_cv]

# Perform cross-validation
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf_results = cross_validate_models(datasets_for_ml, rf_model)
gb_results = cross_validate_models(datasets_for_ml, gb_model)

# Output the results
print("Random Forest Cross-Validation Results:")
for key, value in rf_results.items():
    print(f"{key}: MSE = {value:.2f}")

print("\nGradient Boosting Cross-Validation Results:")
for key, value in gb_results.items():
    print(f"{key}: MSE = {value:.2f}")

# Generate predictions for both models
rf_predictions = rf_optimized_model.predict(X_test_srm22_ml)
gb_predictions = gb_optimized_model.predict(X_test_srm22_ml)

# Plotting Actual vs. Predicted values for Random Forest model
plt.figure(figsize=(10, 5))
plt.scatter(X_test_srm22_ml, y_test_srm22_ml, color='black', label='Actual data')
plt.scatter(X_test_srm22_ml, rf_predictions, color='green', label='RF Predicted data')
plt.title('Random Forest: Actual vs. Predicted Registrations')
plt.xlabel('Time Index')
plt.ylabel('Registrations')
plt.legend()
plt.show()

# Plotting Actual vs. Predicted values for Gradient Boosting model
plt.figure(figsize=(10, 5))
plt.scatter(X_test_srm22_ml, y_test_srm22_ml, color='black', label='Actual data')
plt.scatter(X_test_srm22_ml, gb_predictions, color='blue', label='GB Predicted data')
plt.title('Gradient Boosting: Actual vs. Predicted Registrations')
plt.xlabel('Time Index')
plt.ylabel('Registrations')
plt.legend()
plt.show()
