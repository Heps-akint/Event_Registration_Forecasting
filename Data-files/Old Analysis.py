import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load the data from each file
d19 = pd.read_csv('D19.csv', skiprows=1)
d21 = pd.read_csv('D21.csv', skiprows=1)
gp21 = pd.read_csv('GP21.csv', skiprows=1)
mse21 = pd.read_csv('MSE21.csv', skiprows=1)
np21 = pd.read_csv('NP21.csv', skiprows=1)
srm22 = pd.read_csv('SRM22.csv', skiprows=1)
srm23 = pd.read_csv('SRM23.csv', skiprows=1)
d21a = pd.read_csv('D21A.csv', skiprows=1)
gp21a = pd.read_csv('GP21A.csv', skiprows=1)

# Preview the data to understand its structure
data_previews = {
    'D19': d19.head(),
    'D21': d21.head(),
    'GP21': gp21.head(),
    'MSE21': mse21.head(),
    'NP21': np21.head(),
    'SRM22': srm22.head(),
    'SRM23': srm23.head(),
    'D21A': d21a.head(),
    'GP21A': d21a.head()
}

print(data_previews)

# Standardize column names and convert 'Created Date' to datetime format for each dataframe
dataframes = [d19, d21, gp21, mse21, np21, srm22, srm23, d21a, gp21a]
for df in dataframes:
    df.columns = ['BookingReference', 'CreatedDate', 'Reference', 'AttendeeStatus', 'Attended']
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], dayfirst=True)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# Sort by 'CreatedDate'
combined_df_sorted = combined_df.sort_values(by='CreatedDate')

# Generate a cumulative count of registrations over time
combined_df_sorted['CumulativeRegistrations'] = combined_df_sorted.groupby('AttendeeStatus').cumcount() + 1

# Correct cumulative calculation: we need the overall cumulative count, not grouped by 'AttendeeStatus'
combined_df_sorted['OverallCumulativeRegistrations'] = range(1, len(combined_df_sorted) + 1)

# Display the first few rows of the sorted and corrected cumulative count dataframe
combined_df_sorted.head(), combined_df_sorted['CreatedDate'].min(), combined_df_sorted['CreatedDate'].max()


# Plot cumulative registration trends over time for each individual file
plt.figure(figsize=(14, 10), dpi=100)

# Individual dataframes and their names for legend
dfs = [d19, d21, gp21, mse21, np21, srm22, srm23, d21a, gp21a]
names = ['D19', 'D21', 'GP21', 'MSE21', 'NP21', 'SRM22', 'SRM23', 'D21A', 'GP21A']

for df, name in zip(dfs, names):
    # Sort by 'CreatedDate'
    df_sorted = df.sort_values(by='CreatedDate')
    # Generate a cumulative count of registrations over time
    df_sorted['CumulativeRegistrations'] = range(1, len(df_sorted) + 1)
    # Plot
    plt.plot(df_sorted['CreatedDate'], df_sorted['CumulativeRegistrations'], marker='o', linestyle='-', markersize=2, label=name)

# Enhance the plot
plt.title('Cumulative Registration Trends Over Time by Conference')
plt.xlabel('Date')
plt.ylabel('Cumulative Registrations')
plt.legend()
plt.grid(True)

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

def plot_cumulative_trend(file_name):
    """
    Plots the cumulative registration trend for a specified file.
    
    Parameters:
    - file_name: The name of the file (as a string) to plot. Options are 'D19', 'D21', 'GP21', 'MSE21', 'NP21', 'SRM22', 'SRM23', 'D21A', 'GP21A'.
    """
    # Map the file name to the corresponding dataframe
    file_to_df = {
        'D19': d19,
        'D21': d21,
        'GP21': gp21,
        'MSE21': mse21,
        'NP21': np21,
        'SRM22': srm22,
        'SRM23': srm23,
        'D21A': d21a,
        'GP21A': gp21a
    }
    
    # Get the dataframe for the specified file
    df = file_to_df.get(file_name)
    
    if df is None:
        print(f"No data found for file name: {file_name}. Please choose from 'D19', 'D21', 'GP21', 'MSE21', 'NP21', 'SRM22', 'SRM23', 'D21A', 'GP21A'.")
        return
    
    # Sort by 'CreatedDate'
    df_sorted = df.sort_values(by='CreatedDate')
    # Generate a cumulative count of registrations over time
    df_sorted['CumulativeRegistrations'] = range(1, len(df_sorted) + 1)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(df_sorted['CreatedDate'], df_sorted['CumulativeRegistrations'], marker='o', linestyle='-', markersize=2)
    plt.title(f'Cumulative Registrations for {file_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Registrations')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_cumulative_trend('D19')
# Note: You can call this function with any of the specified file names to plot the respective data.

plot_cumulative_trend('SRM22')
