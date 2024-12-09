import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path to your CSV file
input_csv_path = "data/cleandata/merged_output.csv"

# Load the data
data = pd.read_csv(input_csv_path)

# Split the data (80% train, 20% validation)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create directories for train and validation if they don't exist

# Save the datasets
train_data.to_csv("data/train/cleandata_train.csv", index=False)
val_data.to_csv("data/val/clean_data_val.csv", index=False)

print("Data has been successfully split and saved!")
