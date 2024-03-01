# load the data from the uploaded CSV file to understand its structure and content.
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load the CSV file
file_path = 'D21_cumulative.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

import matplotlib.pyplot as plt

# Plotting the data without data point markers for a smoother curve
plt.figure(figsize=(10, 6))
plt.plot(data['Days Since Start'], data['Cumulative Registrations'], linestyle='-', color='blue')
plt.title('Cumulative Registrations Over Time (Curve)')
plt.xlabel('Days Since Start')
plt.ylabel('Cumulative Registrations')
plt.grid(True)
plt.show()

# To identify the points of interest, we need to calculate the gradient of the curve.
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

def calculate_average_gradient_and_error(file_path, ranges):
    """
    Calculates the average gradient and its error for specified sections of a curve
    represented in a CSV file with columns 'Days Since Start' and 'Cumulative Registrations'.
    
    Parameters:
    - file_path: Path to the CSV file.
    - ranges: A list of tuples, each representing a range (start_day, end_day) for which
              the gradient and its error are to be calculated.
    
    Returns:
    - A tuple containing the average gradient and the standard error of the mean (SEM) for the calculated gradients.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Calculate gradients and errors for the specified ranges
    gradients = []
    std_errors = []
    for start_day, end_day in ranges:
        # Filtering the data for the specified range
        section = data[(data['Days Since Start'] >= start_day) & (data['Days Since Start'] <= end_day)]
        
        # Linear regression to calculate the gradient (slope) and its standard error
        slope, intercept, r_value, p_value, std_err = linregress(section['Days Since Start'], section['Cumulative Registrations'])
        
        gradients.append(slope)
        std_errors.append(std_err)
    
    # Calculate the average gradient
    average_gradient = sum(gradients) / len(gradients)

    # Calculate the standard error of the mean (SEM) for the gradients
    sem = (sum([se**2 for se in std_errors]) / len(std_errors))**0.5

    return average_gradient, sem

# Example usage of the function with the current file and specified ranges
#SRM22 speficied ranges
#specified_ranges = [(14, 43), (51, 70)]
#NP21 specified ranges
#specified_ranges = [(20, 338), (343,387)]
#SRM23 specified ranges
#specified_ranges = [(0, 25), (40, 105), (127, data['Days Since Start'].iloc[-1])]
#MSE21 spefified ranges
#specified_ranges = [(7, 16), (17, 27), (38, 42)]
#D19 spefified ranges
#specified_ranges = [(0, 108)]
#GP21 specified ranges
#specified_ranges = [(2, 130)]
#D21 specified ranges
specified_ranges = [(0, 318)]
#SRM23A specified ranges
#specified_ranges = [(0, 25), (40, 105)]

print(calculate_average_gradient_and_error(file_path, specified_ranges))
