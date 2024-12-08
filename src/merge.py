import pandas as pd
import glob

# Define the path to the CSV files
# Use a wildcard to match all CSV files in a directory (change the path as needed)
csv_files = glob.glob('data/rawdata/*.csv')

# Initialize an empty list to store data frames
dfs = []

# Loop over each CSV file and read them into a list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Merge all dataframes based on Household_ID
# The on parameter specifies the column to merge on (Household_ID)
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='household_ID', how='inner')

# Remove rows with missing values (NaN)
merged_df_cleaned = merged_df.dropna()

# Save the merged and cleaned data to a new CSV file
merged_df_cleaned.to_csv('data/cleandata/merged_cleaned_data.csv', index=False)

# Optionally, print the first few rows of the merged and cleaned data
print(merged_df_cleaned.head())