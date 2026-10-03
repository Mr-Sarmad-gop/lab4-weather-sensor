import numpy as np
import pandas as pd
from scipy import stats

# Input data
temp_data = [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23, 22, 21, 23, 24, 25]
df = pd.DataFrame({
    'Hour': range(1, 16),
    'Temp_raw': temp_data
})

# (a) Stats BEFORE cleaning
raw_clean = df['Temp_raw'].dropna()
print("Stats BEFORE cleaning:")
print(f"Mean: {raw_clean.mean():.2f} | Std: {raw_clean.std():.2f}")
print(f"Skewness: {stats.skew(raw_clean):.2f} | Kurtosis: {stats.kurtosis(raw_clean):.2f}")
print("-" * 50)

# (b) Linear Interpolation for missing values
df['Temp_clean'] = df['Temp_raw'].interpolate(method='linear')
print(f"After interpolation, missing values: {df['Temp_clean'].isnull().sum()}")

# (c) Detect Outlier using Z-score threshold of 2
z_scores_temp = np.abs(stats.zscore(df['Temp_clean']))
outlier_mask = z_scores_temp > 2

outlier_hours = df[outlier_mask]['Hour'].tolist()
outlier_values = df[outlier_mask]['Temp_clean'].tolist()

if outlier_hours:
    for hour, val in zip(outlier_hours, outlier_values):
        print(f"Z-score(>2) outlier detected at Hour: {hour} (value: {val})")
    
    # Replace outlier with median of non-outliers
    median_val = df.loc[~outlier_mask, 'Temp_clean'].median()
    df.loc[outlier_mask, 'Temp_clean'] = median_val
    print(f"Replaced with column median: {median_val}")

print("-" * 50)

# (d) Z-score standardization on cleaned temperature
df['Temp_Zscore'] = stats.zscore(df['Temp_clean'])

print("Before/After comparison (first 8 rows):")
print(df[['Hour', 'Temp_raw', 'Temp_clean', 'Temp_Zscore']].head(8).to_string(index=False))

print("\nStats AFTER cleaning:")
print(f"Mean: {df['Temp_clean'].mean():.2f} | Std: {df['Temp_clean'].std():.2f}")