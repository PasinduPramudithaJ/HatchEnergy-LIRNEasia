import os
import pandas as pd

def sort_csv_files_in_folder(folder_path, sort_column):
    """
    Sort all CSV files in the given folder by the specified column and save them with their existing names.

    :param folder_path: Path to the folder containing the CSV files.
    :param sort_column: The column name by which to sort the CSV files.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Check if the specified column exists
            if sort_column not in df.columns:
                print(f"Column '{sort_column}' not found in {csv_file}. Skipping this file.")
                continue

            # Sort the DataFrame by the specified column
            sorted_df = df.sort_values(by=sort_column)

            # Save the sorted DataFrame back to the same file
            sorted_df.to_csv(file_path, index=False)
            print(f"Sorted and saved: {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# Example usage
folder_path = "./data/rawdata/"  # Replace with the path to your folder
sort_column = "household_ID"          # Replace with the column you want to sort by
sort_csv_files_in_folder(folder_path, sort_column)