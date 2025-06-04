import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Get the current directory and construct the correct path to train.csv
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_csv_path = os.path.join(current_dir, 'train.csv')

# Read the dataset
df = pd.read_csv(train_csv_path)

# Create output directory for plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(output_dir, exist_ok=True)

def analyze_variable_relationships():
    # 1. Null Value Analysis
    print("\n1. Null Value Analysis")
    print("====================")
    
    # Calculate null percentages for each column
    null_percentages = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({
        'Column': null_percentages.index,
        'Null Percentage': null_percentages.values
    }).sort_values('Null Percentage', ascending=False)
    
    print("\nNull Value Percentages by Column:")
    print(null_df)
    
    # Plot null value percentages
    plt.figure(figsize=(12, 6))
    sns.barplot(data=null_df, x='Null Percentage', y='Column')
    plt.title('Percentage of Null Values by Column')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'null_values_percentage.png'))
    plt.close()
    
    # 2. Correlation Analysis with Null Value Information
    print("\n2. Correlation Analysis")
    print("=====================")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix using pairwise complete observations
    corr_matrix = numeric_df.corr(method='pearson', min_periods=1)
    
    # Create a mask for missing values in correlation matrix
    missing_matrix = numeric_df.isnull()
    missing_counts = missing_matrix.sum()
    
    # Plot correlation heatmap with null value information
    plt.figure(figsize=(15, 12))
    
    # Create main correlation heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    
    plt.title('Correlation Heatmap of Numeric Variables\n(Including Null Values)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Strong Correlations Analysis with Null Value Context
    print("\n3. Strong Correlations (|correlation| > 0.5)")
    print("========================================")
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                # Calculate number of complete pairs
                complete_pairs = len(numeric_df[[var1, var2]].dropna())
                strong_correlations.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Correlation': corr_matrix.iloc[i, j],
                    'Complete Pairs': complete_pairs,
                    'Total Rows': len(df),
                    'Complete Pairs %': (complete_pairs / len(df)) * 100
                })
    
    strong_corr_df = pd.DataFrame(strong_correlations)
    if not strong_corr_df.empty:
        print("\nStrong Correlations with Null Value Context:")
        print(strong_corr_df.sort_values('Correlation', ascending=False))
        
        # Plot scatter plots for strongly correlated variables
        for _, row in strong_corr_df.iterrows():
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            sns.scatterplot(data=df, x=row['Variable 1'], y=row['Variable 2'])
            
            # Add null value information to title
            title = f'Relationship between {row["Variable 1"]} and {row["Variable 2"]}\n'
            title += f'Correlation: {row["Correlation"]:.2f}\n'
            title += f'Complete Pairs: {row["Complete Pairs"]} ({row["Complete Pairs %"]:.1f}% of total)'
            
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'scatter_{row["Variable 1"]}_{row["Variable 2"]}.png'))
            plt.close()
    
    # 4. Impact Analysis with Null Value Consideration
    print("\n4. Impact Analysis")
    print("================")
    
    # Calculate impact scores for each variable
    impact_scores = {}
    for col in numeric_df.columns:
        # Calculate average absolute correlation
        impact_scores[col] = {
            'Impact Score': abs(corr_matrix[col]).mean(),
            'Null Percentage': null_percentages[col],
            'Complete Observations': len(df) - missing_counts[col]
        }
    
    impact_df = pd.DataFrame(impact_scores).T
    impact_df = impact_df.sort_values('Impact Score', ascending=False)
    
    print("\nVariable Impact Scores with Null Value Context:")
    print(impact_df)
    
    # Plot impact scores with null value information
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    ax = sns.barplot(data=impact_df.reset_index(), x='Impact Score', y='index')
    
    # Add null percentage as text on bars
    for i, (_, row) in enumerate(impact_df.iterrows()):
        ax.text(row['Impact Score'], i, 
                f'Null: {row["Null Percentage"]:.1f}%\nComplete: {row["Complete Observations"]}',
                va='center')
    
    plt.title('Variable Impact Scores with Null Value Information')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'impact_scores.png'))
    plt.close()
    
    # 5. Statistical Tests with Null Value Context
    print("\n5. Statistical Tests for Relationships")
    print("==================================")
    
    # Perform statistical tests for each pair of variables
    test_results = []
    for i in range(len(numeric_df.columns)):
        for j in range(i+1, len(numeric_df.columns)):
            var1 = numeric_df.columns[i]
            var2 = numeric_df.columns[j]
            
            # Get complete pairs for this variable pair
            valid_data = numeric_df[[var1, var2]].dropna()
            
            if len(valid_data) > 0:
                # Calculate correlation and p-value
                corr, p_value = stats.pearsonr(valid_data[var1], valid_data[var2])
                
                test_results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Correlation': corr,
                    'P-value': p_value,
                    'Significant': p_value < 0.05,
                    'Complete Pairs': len(valid_data),
                    'Complete Pairs %': (len(valid_data) / len(df)) * 100
                })
    
    test_df = pd.DataFrame(test_results)
    print("\nStatistical Test Results with Null Value Context (Top 10 by absolute correlation):")
    print(test_df.sort_values('Correlation', key=abs, ascending=False).head(10))
    
    # 6. Pair Plots for Top Impact Variables with Null Information
    print("\n6. Pair Plots for Top Impact Variables")
    print("===================================")
    
    # Select top 5 variables by impact score
    top_vars = impact_df.index[:5].tolist()
    
    # Create pair plot with null value information
    plt.figure(figsize=(15, 15))
    pair_plot = sns.pairplot(numeric_df[top_vars], diag_kind='kde')
    
    # Add null value information to the title
    null_info = "\n".join([f"{var}: {null_percentages[var]:.1f}% null" for var in top_vars])
    pair_plot.fig.suptitle(f'Pair Plots for Top 5 Impact Variables\nNull Value Percentages:\n{null_info}', y=1.02)
    
    plt.savefig(os.path.join(output_dir, 'pair_plot_top_variables.png'))
    plt.close()

if __name__ == "__main__":
    analyze_variable_relationships()
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.") 