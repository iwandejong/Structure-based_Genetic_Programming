import pandas as pd
import numpy as np
from scipy import stats

datasets = ["197_cpu_act.tsv", "227_cpu_small.tsv"]

for dataset in datasets:
  df = pd.read_csv(dataset, sep="\t")

  # Remove duplicates
  df = df.drop_duplicates()

  # Remove outliers
  threshold_z = 2
  outlier_mask = np.zeros(len(df), dtype=bool)

  for col in df.columns:
    z = np.abs(stats.zscore(df[col]))
    outlier_mask = outlier_mask | (z > threshold_z)

  # Keep only the non-outlier rows
  df_clean = df[~outlier_mask]

  target = df_clean["target"].copy()
  df_clean = df_clean.drop(columns=["target"])

  # Drop columns with more than 25% zero values
  zero_inflated_columns = df_clean.columns[df_clean.isin([0.0]).sum() > (0.25 * len(df_clean))]
  df_clean = df_clean.drop(columns=zero_inflated_columns)

  df = pd.concat([df_clean, target], axis=1)

  from sklearn.preprocessing import PowerTransformer
  pt = PowerTransformer(method='yeo-johnson')

  df_preTransform = df.copy()

  # Apply Yeo Johnson transformation to all columns
  for col in df.columns:
    df[col] = pt.fit_transform(df[[col]])

  # Normalise the data using column-wise Min-Max scaling
  for col in df.columns:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val)

  # Save tsv
  df.to_csv(f"{dataset.split(".")[0]}_cleaned.tsv", sep="\t", index=False)
