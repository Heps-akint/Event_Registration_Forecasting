# load the data from the uploaded CSV file to understand its structure and content.
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'MSE21_cumulative.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Plotting the data without data point markers for a smoother curve
plt.figure(figsize=(10, 6))
plt.plot(data['Days Since Start'], data['Cumulative Registrations'], linestyle='-', color='blue')
plt.title('Cumulative Registrations Over Time (Curve)')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Registrations')
plt.grid(True)
plt.show()

# Identify the points of interest, we need to calculate the gradient of the curve.
# We're looking for points where there is a sudden rise in registrations followed by a steady, low gradient.

# Calculate the gradient of the cumulative registrations
gradient = np.gradient(data['Cumulative Registrations'].values)

# Find the indices where there's a significant change in gradient
# Define a "sudden rise" as a significant increase in gradient, followed by a "steady, low gradient"

# Identifying significant changes might require comparing gradients and finding points where a sudden rise is followed by a steadier phase
# This could be done by identifying peaks in the gradient, then checking for periods of lower gradients after these peaks.

# visually identify these changes based on the plotted gradient changes.
plt.figure(figsize=(12, 6))

# Plotting the gradient to help identify points of interest
plt.plot(data['Days Since Start'], gradient, label='Gradient of Registrations', color='green')
plt.title('Gradient of Cumulative Registrations')
plt.xlabel('Days Since Start')
plt.ylabel('Gradient')
plt.axhline(y=0, color='black', linestyle='--')  # Reference line at zero gradient
plt.legend()
plt.grid(True)
plt.show()

# Based on observation, highlight the correct parts of the curve we are interested in.

# Defining the ranges to highlight based on user input
highlight_ranges = [(7, 16), (17, 27), (38, 42)]

# Plotting the original curve
plt.figure(figsize=(12, 7))
plt.plot(data['Days Since Start'], data['Cumulative Registrations'], linestyle='-', color='blue', label='Cumulative Registrations')

# Highlighting the specified ranges
for start, end in highlight_ranges:
    plt.fill_betweenx(data['Cumulative Registrations'], start, end, color='red', alpha=0.3)

plt.title('Cumulative Registrations with Highlighted Areas')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Registrations')
plt.grid(True)
plt.legend()
plt.show()

# Function to calculate the gradient and its error for specified sections of the curve
def calculate_gradients_and_errors(data, ranges):
    results = []
    for start_day, end_day in ranges:
        # Filtering the data for the specified range
        section = data[(data['Days Since Start'] >= start_day) & (data['Days Since Start'] <= end_day)]
        
        # Performing linear regression on the section to calculate the gradient (slope) and its standard error
        slope, intercept, r_value, p_value, std_err = linregress(section['Days Since Start'], section['Cumulative Registrations'])
        
        results.append((slope, std_err))
    
    return results

# Calculate gradients and errors for the highlighted sections
gradients_and_errors = calculate_gradients_and_errors(data, highlight_ranges)

print(gradients_and_errors)

# To calculate the average of the gradients and the error for the calculated average, we will use the formula for the standard error of the mean.

# Extracting gradients and their standard errors
gradients = [result[0] for result in gradients_and_errors]
std_errors = [result[1] for result in gradients_and_errors]

# Calculating the average gradient
average_gradient = sum(gradients) / len(gradients)

# Calculating the standard error of the mean (SEM) for the gradients
# The SEM is calculated as the square root of the sum of squared standard errors divided by the number of observations
sem = (sum([se**2 for se in std_errors]) / len(std_errors))**0.5

print(average_gradient, sem)
