import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "data/metadata.csv"

df = pd.read_csv(filename)

# df['filename_index'] = df['filename_index'].astype(str)
df = df[df['filename_index'].str.contains('train', case=False, na=False)]

# Select all columns except 'filename_index'
columns_to_plot = df.drop(columns=['filename_index']).columns

# Plot category counts
# for col in columns_to_plot:
#     plt.figure()
#     df[col].value_counts().plot(kind='bar', edgecolor='black')
#     plt.title(f'Value counts for {col}')
#     plt.xlabel(col)
#     plt.ylabel('Count')
#     plt.show()
# pass

# Plot correlations
target_col = 'taxonID_index'
df = df.drop(columns=['filename_index'])

# Make a copy
df_encoded = df.copy()

# Encode all columns (since they're categorical)
for col in df_encoded.columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Compute weights based on class frequencies of the target
class_counts = df_encoded[target_col].value_counts()
weights = df_encoded[target_col].map(class_counts).astype(float)

def weighted_corr(x, y, w):
    """Weighted Pearson correlation for categorical codes."""
    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    cov_xy = np.average((x - w_mean_x) * (y - w_mean_y), weights=w)
    std_x = np.sqrt(np.average((x - w_mean_x)**2, weights=w))
    std_y = np.sqrt(np.average((y - w_mean_y)**2, weights=w))
    return cov_xy / (std_x * std_y)

# Calculate weighted correlations for all other columns
weighted_corrs = {}
for col in df_encoded.columns:
    if col != target_col:
        weighted_corrs[col] = weighted_corr(df_encoded[col], df_encoded[target_col], weights)

# Convert to Series for easy viewing
weighted_corrs = pd.Series(weighted_corrs).sort_values(key=abs, ascending=False)

# Weighted mean correlation (absolute values)
weighted_mean_corr = weighted_corrs.abs().mean()

print("Weighted correlations with target:")
print(weighted_corrs)
print(f"\nWeighted mean correlation: {weighted_mean_corr:.4f}")

pass