# HEATMAPS AND MISSING VALUES

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the current directory and construct the correct path to train.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
train_csv_path = os.path.join(current_dir, '..', 'train.csv')

# Read the dataset
df = pd.read_csv(train_csv_path)

# Define common null value representations
null_values = [
    'null', 'NULL', 'Null',
    'none', 'None', 'NONE',
    'nan', 'NaN', 'NAN',
    'na', 'NA', 'N/A', 'n/a',
    'unknown', 'Unknown', 'UNKNOWN',
    'missing', 'Missing', 'MISSING',
    'undefined', 'Undefined', 'UNDEFINED',
    '?', '??', '???',
    '-', '--', '---',
    'empty', 'Empty', 'EMPTY'
]

# Function to analyze missing values
def analyze_missing_values(series):
    # Get actual null values
    null_mask = series.isnull()
    null_count = null_mask.sum()
    
    # Get null-like values (excluding actual nulls)
    if series.dtype == 'object':
        null_like_mask = series.astype(str).str.lower().isin([x.lower() for x in null_values])
        # Exclude actual nulls from null-like count
        null_like_count = (null_like_mask & ~null_mask).sum()
    else:
        null_like_count = 0
    
    return null_count + null_like_count

# Calculate missing values for each column
missing_counts = {}
for column in df.columns:
    missing_counts[column] = analyze_missing_values(df[column])

# Convert to percentages
missing_percentages = pd.Series(missing_counts) / len(df) * 100

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Create correlation matrix with pairwise complete observations
correlation_matrix = numeric_df.corr(method='pearson', min_periods=1)

# Set up the figure size for better visualization
plt.figure(figsize=(15, 10))

# Create subplot for correlation heatmap
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',  # Format correlation values to 2 decimal places
            square=True,  # Make the plot square
            linewidths=0.5)  # Add lines between cells

plt.title("Correlation Heatmap\n(Using pairwise complete observations)", pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Create subplot for missing values
plt.subplot(1, 2, 2)
missing_percentages.sort_values(ascending=False).plot(kind='bar')
plt.title("Percentage of Missing Values by Column", pad=20)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Percentage of Missing Values")
plt.ylim(0, 100)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Print detailed statistics about missing values
print("\nMissing Values Analysis:")
print("=======================")
print("\nPercentage of missing values in each column:")
for col, percentage in missing_percentages.sort_values(ascending=False).items():
    print(f"{col}: {percentage:.2f}%")

print("\nCorrelation Analysis Notes:")
print("========================")
print("1. Correlations are calculated using pairwise complete observations")
print("2. This means that for each pair of variables, only the rows where both values are present are used")
print("3. This approach preserves the original data structure and doesn't make assumptions about missing values")
print("4. The correlation matrix shows the relationships between variables where data is available")

# Show the plots
plt.show()