import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
# Use the Excel file from the workspace
file_path = "/workspaces/statsniella/Stats.xlsx"

if os.path.exists(file_path):
    data = pd.read_excel(file_path)
    data.dropna(inplace=True) # Remove empty rows
    print(f"SUCCESS: Data loaded from {file_path}")
    print(f"Columns found: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    
    # Use Math_Score as "Before" and English_Score as "After"
    kangkong = pd.DataFrame({
        'Before': data['Math_Score'],
        'After': data['English_Score']
    })
else:
    # Generate consistent dummy data for testing if file is missing
    np.random.seed(42)
    kangkong = pd.DataFrame({
        'Before': np.random.normal(loc=15, scale=3, size=30),
        'After': np.random.normal(loc=19, scale=3.5, size=30)
    })
    print("WARNING: File not found. Using generated dummy data.")

# Calculate Difference
kangkong['Difference'] = kangkong['After'] - kangkong['Before']

# Reshape for Boxplot (Long Format)
kangkong_long = kangkong.melt(
    value_vars=['Before', 'After'], 
    var_name='Time', 
    value_name='Score'
)

# ==========================================
# 2. STATISTICAL TESTS
# ==========================================
# Normality (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(kangkong['Difference'])

# Paired T-Test (Parametric)
t_stat, t_p = stats.ttest_rel(kangkong['After'], kangkong['Before'])

# Wilcoxon Signed-Rank (Non-Parametric)
wilcox_stat, wilcox_p = stats.wilcoxon(kangkong['After'], kangkong['Before'])

# ==========================================
# 3. GENERATE PLOTS
# ==========================================
# Set a professional theme
sns.set_theme(style="whitegrid", palette="muted")

# Create a 2x2 Grid of Plots with better spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Paired Analysis: Before vs. After', fontsize=18, fontweight='bold', y=0.995)

# --- PLOT A: Scatter Plot (Correlation & Growth) ---
# Shows relationship between Before and After scores
sns.regplot(
    x='Before', y='After', data=kangkong, ax=axes[0, 0],
    scatter_kws={'s': 60, 'alpha': 0.7, 'color': 'darkblue'},
    line_kws={'color': 'red', 'linewidth': 2, 'label': 'Regression Line'}
)
# Add y=x line (No Change Line)
lims = [
    np.min([axes[0, 0].get_xlim(), axes[0, 0].get_ylim()]),
    np.max([axes[0, 0].get_xlim(), axes[0, 0].get_ylim()])
]
axes[0, 0].plot(lims, lims, 'k--', alpha=0.5, label="No Change (y=x)")
axes[0, 0].set_title("Scatter Plot: Correlation", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Before Score', fontsize=12)
axes[0, 0].set_ylabel('After Score', fontsize=12)
axes[0, 0].legend(loc='best')

# --- PLOT B: Boxplot with Strip Overlay ---
# [Image of box plot anatomy]
# Shows distribution summary (Median, Quartiles) + Individual Points
sns.boxplot(
    x='Time', y='Score', data=kangkong_long, ax=axes[0, 1],
    palette={'Before': '#ff9999', 'After': '#99ff99'}, showfliers=False
)
# Add strip plot to show actual sample size/density
sns.stripplot(
    x='Time', y='Score', data=kangkong_long, ax=axes[0, 1],
    color='black', alpha=0.5, jitter=True
)
axes[0, 1].set_title("Boxplot: Central Tendency", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Time Period', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)

# --- PLOT C: Histogram of Differences ---
# Checks if the improvement/decline is normally distributed
sns.histplot(
    kangkong['Difference'], kde=True, ax=axes[1, 0],
    color='teal', bins=10, edgecolor='black'
)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Difference')
axes[1, 0].set_title("Histogram: Distribution of Differences", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Difference (After - Before)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].legend(loc='best')

# --- PLOT D: QQ Plot (Normality Check) ---
# Visual confirmation of the Shapiro-Wilk test
# Points should hug the red line for data to be Normal
(quantiles, values), (slope, intercept, r) = stats.probplot(kangkong['Difference'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title(f"QQ Plot (Normality Check)\nR²={r**2:.3f}", fontsize=14, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[1, 1].set_ylabel('Sample Quantiles', fontsize=12)
axes[1, 1].get_lines()[0].set_markerfacecolor('purple')
axes[1, 1].get_lines()[0].set_markersize(6.0)

# Adjust layout to prevent overlap with better spacing
plt.tight_layout(rect=[0, 0.01, 1, 0.99], h_pad=3.0, w_pad=3.0)

# Save the plot as PNG
output_file = "/workspaces/statsniella/paired_analysis_plots.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Show the plot window
plt.show()

# ==========================================
# 4. PRINT SUMMARY
# ==========================================
print(f"\n{'='*40}")
print(f"{'STATISTICAL SUMMARY':^40}")
print(f"{'='*40}")

print("\n1. NORMALITY CHECK (Difference)")
print(f"   Shapiro-Wilk p-value: {shapiro_p:.5f}")
if shapiro_p > 0.05:
    print("   -> Data is NORMAL. (Check QQ Plot bottom-right)")
else:
    print("   -> Data is NOT NORMAL. (Check QQ Plot bottom-right)")

print("\n2. HYPOTHESIS TESTS")
print(f"   Paired T-Test p-value:       {t_p:.5f}")
print(f"   Wilcoxon Signed-Rank p-value: {wilcox_p:.5f}")

print("\n3. DESCRIPTIVE")
print(f"   Mean Difference (Growth):     {kangkong['Difference'].mean():.2f}")
print(f"{'='*40}\n")