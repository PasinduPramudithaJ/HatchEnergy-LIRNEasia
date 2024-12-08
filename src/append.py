import pandas as pd
import os

# Paths to the two CSV files
file1 = 'data/append/output3.csv'  # Replace with your first CSV file name
file2 = 'data/append/avg_nonsmart.csv'  # Replace with your second CSV file name

# Ensure the output folder exists
output_folder = 'data/rawdata/'
os.makedirs(output_folder, exist_ok=True)

# Path to save the appended file
output_file = os.path.join(output_folder, 'append.csv')

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Append the DataFrames
appended_df = pd.concat([df1, df2], ignore_index=True)

# Save the result to a new CSV file
appended_df.to_csv(output_file, index=False)

print(f'Files have been appended and saved to {output_file}')
