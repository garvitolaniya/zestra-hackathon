import pandas as pd
import numpy as np

# Read the train.csv file
df = pd.read_csv('dataset/train.csv')

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

# Function to check for null-like values and verify no double counting
def analyze_missing_values(series):
    # Get actual null values
    null_mask = series.isnull()
    null_count = null_mask.sum()
    
    # Get null-like values (excluding actual nulls)
    if series.dtype == 'object':
        null_like_mask = series.astype(str).str.lower().isin([x.lower() for x in null_values])
        # Exclude actual nulls from null-like count
        null_like_count = (null_like_mask & ~null_mask).sum()
        
        # Get sample of null-like values
        null_like_values = series[null_like_mask & ~null_mask]
        sample_values = null_like_values.value_counts().head(3)
    else:
        null_like_count = 0
        sample_values = pd.Series()
    
    return pd.Series({
        'Null Values': null_count,
        'Null-like Values': null_like_count,
        'Sample Null-like': sample_values
    })

# Analyze each column
analysis_results = {}
for column in df.columns:
    analysis_results[column] = analyze_missing_values(df[column])

# Create summary DataFrame
summary_data = []
for column, results in analysis_results.items():
    row_data = {
        'Column': column,
        'Total Rows': len(df),
        'Null Values': results['Null Values'],
        'Null Percentage': (results['Null Values'] / len(df)) * 100,
        'Null-like Values': results['Null-like Values'],
        'Null-like Percentage': (results['Null-like Values'] / len(df)) * 100,
        'Total Missing': results['Null Values'] + results['Null-like Values'],
        'Total Missing Percentage': ((results['Null Values'] + results['Null-like Values']) / len(df)) * 100
    }
    summary_data.append(row_data)

null_summary = pd.DataFrame(summary_data)
null_summary = null_summary.sort_values('Total Missing', ascending=False)

# Print detailed summary
print("\nDetailed Analysis of Missing Values in Dataset")
print("=============================================")
print(f"Total number of rows in dataset: {len(df)}")
print(f"Total number of columns in dataset: {len(df.columns)}")
print("\nMissing Values Summary by Column:")
print("================================")

for _, row in null_summary.iterrows():
    print(f"\n{row['Column']}:")
    print(f"Total Rows: {row['Total Rows']}")
    print(f"Null Values: {row['Null Values']} ({row['Null Percentage']:.2f}%)")
    print(f"Null-like Values: {row['Null-like Values']} ({row['Null-like Percentage']:.2f}%)")
    print(f"Total Missing: {row['Total Missing']} ({row['Total Missing Percentage']:.2f}%)")
    
    # Print sample of null-like values if they exist
    sample_values = analysis_results[row['Column']]['Sample Null-like']
    if not sample_values.empty:
        print("Sample of null-like values found:")
        for value, count in sample_values.items():
            print(f"  '{value}': {count} occurrences")

# Print total statistics
print("\nOverall Dataset Statistics:")
print("=========================")
total_nulls = null_summary['Null Values'].sum()
total_null_like = null_summary['Null-like Values'].sum()
print(f"Total null values: {total_nulls}")
print(f"Total null-like values: {total_null_like}")
print(f"Total missing values (null + null-like): {total_nulls + total_null_like}")

# Verify no double counting
print("\nVerification of Double Counting:")
print("==============================")
for column in df.columns:
    null_mask = df[column].isnull()
    if df[column].dtype == 'object':
        null_like_mask = df[column].astype(str).str.lower().isin([x.lower() for x in null_values])
        overlap = (null_mask & null_like_mask).sum()
        if overlap > 0:
            print(f"{column}: Found {overlap} values that are both null and null-like") 