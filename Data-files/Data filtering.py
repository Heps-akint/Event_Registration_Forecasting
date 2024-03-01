# Read the uploaded CSV file to understand its structure and the data it contains.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ML stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = 'SRM22.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Correct the dataframe by setting the first row as headers
data.columns = data.iloc[0] # Set the first row as column names
data = data[1:] # Remove the first row from the dataframe

# Convert 'Created Date' to datetime format to facilitate time-based analysis
data['Created Date'] = pd.to_datetime(data['Created Date'], dayfirst=True)

# Prepare data for cumulative registration trends over time
# Count registrations by date and then calculate the cumulative sum.
registration_counts = data.groupby('Created Date').size().reset_index(name='Registrations')
registration_counts['Cumulative Registrations'] = registration_counts['Registrations'].cumsum()

registration_counts.head()

# Plotting the cumulative registration trends
plt.figure(figsize=(12, 6))
plt.plot(registration_counts['Created Date'], registration_counts['Cumulative Registrations'], marker='o', linestyle='-', color='b', markersize=4)
plt.title('Cumulative Registration Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Registrations')
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.tight_layout() # Adjust layout to not cut off labels
plt.show()

# Calculate 'Days Since Start'

# Confirm the registration start date
registration_start_date_confirmed = data['Created Date'].min()

# Calculate the days since the start for each entry
data['Days Since Start'] = (data['Created Date'] - registration_start_date_confirmed).dt.days

# Check for unique values in 'Days Since Start Corrected' to ensure it's not all zeros
unique_days_since_start = data['Days Since Start'].unique()

print(unique_days_since_start, registration_start_date_confirmed)

# Display the first few rows to confirm the transformation
print(data.head())

# Drop all columns except 'Days Since Start' and then calculate cumulative registrations
# Group by 'Days Since Start Corrected' to avoid dropping necessary data for cumulative calculation

# Group by 'Days Since Start Corrected' to get the count of registrations for each day
registrations_per_day_corrected = data.groupby('Days Since Start').size()

# Calculate the cumulative sum of registrations over the corrected days since start
data = registrations_per_day_corrected.cumsum().reset_index()

# Rename columns for clarity
data.columns = ['Days Since Start', 'Cumulative Registrations']

# Display the resulting DataFrame
print(data.head())

# Plot the cumulative registrations against days since start
plt.figure(figsize=(12, 6))
plt.plot(data['Days Since Start'], data['Cumulative Registrations'], marker='o', linestyle='-', color='blue')
plt.title('Cumulative Registrations vs. Days Since Start')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Registrations')
plt.grid(True)
plt.tight_layout()  # Adjust layout to not cut off labels

# Show the plot
plt.show()

# Save the DataFrame with 'Days Since Start' and 'Cumulative Registrations' to a new CSV file
new_csv_path = 'SRM22_cumulative.csv'
data.to_csv(new_csv_path, index=False)

'''
Everything beyond this point is useless now.

Has been deemed out of scopr or not viable
'''

# Provide the path for download
print(new_csv_path)

# Machine Learning

# Preparing the data
X = data[['Days Since Start']]
y = data['Cumulative Registrations']

# Applying polynomial features
degree = 5 # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Training the model
model = LinearRegression()
model.fit(X_poly, y)

# Plotting the model's fit to the data
X_fit = pd.DataFrame({'Days Since Start': np.linspace(X['Days Since Start'].min(), X['Days Since Start'].max(), 100)})
# Predicting across the entire range of X
X_fit_poly = poly_features.transform(X_fit)
y_fit = model.predict(X_fit_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_fit, y_fit, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to Cumulative Registrations')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Registrations')
plt.legend()
plt.grid(True)
plt.show()

# To provide errors we'll calculate the mean squared error (MSE) and the coefficient of determination (R^2) for the model.
# Since we used the entire dataset for training, we'll calculate these metrics on the same dataset.

# Calculating MSE and R^2 for the higher degree model
# Use the original X_poly for predictions to match the y dataset
y_pred = model.predict(X_poly)

# Now, calculating MSE and R^2 using y and y_pred (which have the same length)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(mse, r2)
