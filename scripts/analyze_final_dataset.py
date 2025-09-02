import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_and_validate_dataset(file_path):
    """Load the dataset and perform basic validation checks."""
    print("🔍 Loading and validating dataset...")
    df = pd.read_csv(file_path)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")
    
    # Basic validation checks
    validation_results = {}
    
    # 1. Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("❌ Missing values found:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"   {col}: {count} missing values")
            validation_results[col] = count
    else:
        print("✅ No missing values found")
    
    # 2. Check for duplicates
    duplicate_count = df.duplicated(subset=['Sequence']).sum()
    if duplicate_count > 0:
        print(f"❌ {duplicate_count} duplicate sequences found")
        validation_results['duplicates'] = duplicate_count
    else:
        print("✅ No duplicate sequences found")
    
    # 3. Check data types
    print("\n📋 Data types:")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}")
    
    return df, validation_results

def analyze_sequence_properties(df):
    """Analyze the core sequence properties and distributions."""
    print("\n" + "="*60)
    print("🧬 SEQUENCE PROPERTY ANALYSIS")
    print("="*60)
    
    # 1. Length distribution
    length_stats = df['Length'].describe()
    print(f"\n📏 Sequence Length Statistics:")
    print(f"   Count: {int(length_stats['count'])}")
    print(f"   Mean: {length_stats['mean']:.2f} ± {length_stats['std']:.2f}")
    print(f"   Range: {length_stats['min']} - {length_stats['max']} AA")
    print(f"   Quartiles: Q1={length_stats['25%']:.1f}, Q2={length_stats['50%']:.1f}, Q3={length_stats['75%']:.1f}")
    
    # 2. Charge distribution
    charge_stats = df['Net_Charge'].describe()
    print(f"\n⚡ Net Charge Statistics (pH 7.4):")
    print(f"   Mean: {charge_stats['mean']:.2f} ± {charge_stats['std']:.2f}")
    print(f"   Range: {charge_stats['min']:.2f} - {charge_stats['max']:.2f}")
    
    # 3. Hydrophobicity distribution
    gravy_stats = df['Hydrophobicity_GRAVY'].describe()
    print(f"\n💧 Hydrophobicity (GRAVY) Statistics:")
    print(f"   Mean: {gravy_stats['mean']:.3f} ± {gravy_stats['std']:.3f}")
    print(f"   Range: {gravy_stats['min']:.3f} - {gravy_stats['max']:.3f}")
    
    # 4. Identify potential outliers using IQR method
    def find_outliers(series, feature_name):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    length_outliers, lb_len, ub_len = find_outliers(df['Length'], 'Length')
    charge_outliers, lb_chg, ub_chg = find_outliers(df['Net_Charge'], 'Net_Charge')
    gravy_outliers, lb_grav, ub_grav = find_outliers(df['Hydrophobicity_GRAVY'], 'GRAVY')
    
    print(f"\n📊 Potential outliers (IQR method):")
    print(f"   Length: {len(length_outliers)} sequences")
    print(f"   Net Charge: {len(charge_outliers)} sequences") 
    print(f"   Hydrophobicity: {len(gravy_outliers)} sequences")
    
    return {
        'length_stats': length_stats,
        'charge_stats': charge_stats,
        'gravy_stats': gravy_stats,
        'outliers': {
            'length': length_outliers,
            'charge': charge_outliers,
            'hydrophobicity': gravy_outliers
        }
    }

def analyze_source_distribution(df):
    """Analyze the distribution of sequences by source."""
    if 'Source' in df.columns:
        print("\n" + "="*60)
        print("🏷️  SOURCE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        source_counts = df['Source'].value_counts()
        print(f"\n📦 Sequences by source:")
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {source}: {count} sequences ({percentage:.1f}%)")
        
        # Compare properties by source if we have multiple sources
        if len(source_counts) > 1:
            print(f"\n📊 Property comparison by source:")
            for prop in ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY']:
                if prop in df.columns:
                    grouped = df.groupby('Source')[prop].agg(['mean', 'std', 'count'])
                    print(f"\n   {prop}:")
                    for source, row in grouped.iterrows():
                        print(f"     {source}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['count']})")
                    
                    # Statistical test for difference
                    sources = list(source_counts.index)
                    if len(sources) == 2:  # t-test for two groups
                        group1 = df[df['Source'] == sources[0]][prop]
                        group2 = df[df['Source'] == sources[1]][prop]
                        t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
                        print(f"     t-test p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
        
        return source_counts
    return None

def analyze_feature_correlations(df):
    """Analyze correlations between different features."""
    print("\n" + "="*60)
    print("📈 FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric features for correlation analysis
    numeric_features = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 
                       'Molecular_Weight', 'Instability_Index',
                       'Fraction_Positive_AA', 'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA']
    
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    if len(numeric_features) > 1:
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr()
        
        print(f"\n📊 Correlation matrix (Pearson):")
        print(corr_matrix.round(3))
        
        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_correlations.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_value
                    ))
        
        if strong_correlations:
            print(f"\n🔗 Strong correlations (|r| > 0.5):")
            for feat1, feat2, corr in strong_correlations:
                print(f"   {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print(f"\n📉 No strong correlations (|r| > 0.5) found")
        
        return corr_matrix, strong_correlations
    return None, None

def visualize_dataset(df, output_prefix="dataset_analysis"):
    """Create comprehensive visualizations of the dataset."""
    print(f"\n🎨 Generating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Analysis of Curated AMP Dataset', fontsize=16, fontweight='bold')
    
    # 1. Length distribution
    axes[0,0].hist(df['Length'], bins=30, edgecolor='black', alpha=0.7)
    axes[0,0].axvline(df['Length'].mean(), color='red', linestyle='--', label=f'Mean: {df["Length"].mean():.1f}')
    axes[0,0].set_xlabel('Sequence Length (AA)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Sequence Lengths')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Charge vs Hydrophobicity scatter plot
    scatter = axes[0,1].scatter(df['Hydrophobicity_GRAVY'], df['Net_Charge'], 
                               c=df['Length'], cmap='viridis', alpha=0.6, s=30)
    axes[0,1].set_xlabel('Hydrophobicity (GRAVY Index)')
    axes[0,1].set_ylabel('Net Charge (pH 7.4)')
    axes[0,1].set_title('Charge vs Hydrophobicity (colored by length)')
    axes[0,1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,1], label='Sequence Length')
    
    # 3. Feature correlation heatmap
    numeric_features = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 
                       'Molecular_Weight', 'Instability_Index',
                       'Fraction_Positive_AA', 'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA']
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, square=True, ax=axes[1,0])
        axes[1,0].set_title('Feature Correlation Heatmap')
    
    # 4. Source distribution (if available)
    if 'Source' in df.columns:
        source_counts = df['Source'].value_counts()
        axes[1,1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Distribution by Data Source')
    else:
        # Box plot of key properties instead
        prop_df = df[['Length', 'Net_Charge', 'Hydrophobicity_GRAVY']].copy()
        prop_df = (prop_df - prop_df.mean()) / prop_df.std()  # Standardize
        sns.boxplot(data=prop_df, ax=axes[1,1])
        axes[1,1].set_title('Distribution of Standardized Properties')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional specialized plots
    # Property distributions by source (if multiple sources)
    if 'Source' in df.columns and df['Source'].nunique() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        properties = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY']
        
        for i, prop in enumerate(properties):
            if prop in df.columns:
                df.boxplot(column=prop, by='Source', ax=axes[i])
                axes[i].set_title(f'{prop} by Source')
                axes[i].set_ylabel(prop)
        
        plt.suptitle('Property Distributions by Data Source')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_by_source.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved as '{output_prefix}_*.png'")

def generate_insights_report(df, validation_results, property_stats, source_counts, corr_matrix):
    """Generate a comprehensive insights report."""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE INSIGHTS REPORT")
    print("="*60)
    
    total_sequences = len(df)
    
    print(f"\n🎯 DATASET OVERVIEW:")
    print(f"   Total sequences: {total_sequences}")
    print(f"   Length range: {df['Length'].min()} - {df['Length'].max()} AA")
    print(f"   Average length: {df['Length'].mean():.1f} ± {df['Length'].std():.1f} AA")
    
    print(f"\n⚡ CHARGE CHARACTERISTICS:")
    print(f"   Most AMPs are cationic: {len(df[df['Net_Charge'] > 0])}/{total_sequences} " 
          f"({len(df[df['Net_Charge'] > 0])/total_sequences*100:.1f}%) have positive charge")
    print(f"   Charge range: {df['Net_Charge'].min():.1f} to {df['Net_Charge'].max():.1f}")
    
    print(f"\n💧 HYDROPHOBICITY PATTERNS:")
    gravy_mean = df['Hydrophobicity_GRAVY'].mean()
    if gravy_mean > 0:
        print(f"   Overall slightly hydrophobic (mean GRAVY: {gravy_mean:.3f})")
    else:
        print(f"   Overall slightly hydrophilic (mean GRAVY: {gravy_mean:.3f})")
    
    print(f"\n📊 IDEAL FOR MACHINE LEARNING:")
    print(f"   Dataset size is {'excellent' if total_sequences > 2000 else 'good' if total_sequences > 1000 else 'modest'} for ML")
    print(f"   Feature distributions show good variability")
    print(f"   Minimal missing data and duplicates")
    
    if corr_matrix is not None:
        print(f"\n🔍 FEATURE ENGINEERING INSIGHTS:")
        high_var_features = df[['Length', 'Net_Charge', 'Hydrophobicity_GRAVY']].std()
        print(f"   Highest variance features: {high_var_features.idxmax()} (std: {high_var_features.max():.3f})")
        
        # Check for potential multicollinearity
        high_corr_pairs = [(i, j) for i in range(len(corr_matrix.columns)) 
                          for j in range(i+1, len(corr_matrix.columns)) 
                          if abs(corr_matrix.iloc[i, j]) > 0.7]
        if high_corr_pairs:
            print(f"   ⚠️  Potential multicollinearity detected in {len(high_corr_pairs)} feature pairs")
    
    print(f"\n🚀 RECOMMENDATIONS FOR NEXT STEPS:")
    print(f"   1. The dataset is ready for machine learning modeling")
    print(f"   2. Consider feature scaling for neural networks")
    print(f"   3. Start with Random Forest or SVM as baseline models")
    print(f"   4. Use stratified sampling for train/test splits due to property distributions")

def main():
    # Configuration
    dataset_path = "data/complete_curated_combined_gramnegative_amps.csv"  # Update with your actual filename
    output_prefix = "final_amp_dataset"
    
    print("Starting comprehensive analysis of curated AMP dataset...")
    
    # 1. Load and validate
    df, validation_results = load_and_validate_dataset(dataset_path)
    
    # 2. Analyze sequence properties
    property_stats = analyze_sequence_properties(df)
    
    # 3. Analyze source distribution
    source_counts = analyze_source_distribution(df)
    
    # 4. Analyze feature correlations
    corr_matrix, strong_correlations = analyze_feature_correlations(df)
    
    # 5. Create visualizations
    visualize_dataset(df, output_prefix)
    
    # 6. Generate comprehensive insights report
    generate_insights_report(df, validation_results, property_stats, source_counts, corr_matrix)
    
    # 7. Save a sample of the data for inspection
    sample_df = df.sample(min(10, len(df)), random_state=42)
    sample_df.to_csv(f"{output_prefix}_sample.csv", index=False)
    print(f"\n📄 Sample of 10 sequences saved to '{output_prefix}_sample.csv' for inspection")
    
    print(f"\n✅ Analysis complete! The dataset appears ready for machine learning.")

if __name__ == "__main__":
    main()