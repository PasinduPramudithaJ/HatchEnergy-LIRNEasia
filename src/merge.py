import pandas as pd
import glob
import os

def merge_csv_files(input_folder: str, output_file: str):
    """
    Merges all CSV files in a folder based on the 'household_ID' column.
    
    Args:
    - input_folder (str): Path to the folder containing CSV files.
    - output_file (str): Path to save the merged CSV file.
    """
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Merge CSV files
    merged_df = None
    for file in csv_files:
        df = pd.read_csv(file)
        
        # Drop any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if 'household_ID' not in df.columns:
            print(f"File {file} skipped because 'household_ID' column is missing.")
            continue
        
        # Set 'household_ID' as the index for merging
        df.set_index('household_ID', inplace=True)

        # Merge with the previous DataFrame
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, how='outer', left_index=True, right_index=True)
    
    if merged_df is not None:
        # Reset the index to save 'household_ID' as a column
        merged_df.reset_index(inplace=True)
        
        # Save the merged CSV
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV file saved to {output_file}")
    else:
        print("No files were merged.")

# Specify the folder with input CSV files and output file
input_folder = "./data/rawdata/"  # Folder containing the CSV files
output_file = "./data/cleandata/merged_output.csv"  # Path to save the merged CSV

# Call the function
merge_csv_files(input_folder, output_file)
