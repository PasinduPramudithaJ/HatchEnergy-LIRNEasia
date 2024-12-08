import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path to your CSV file
input_csv_path = "data/rawdata/non_smart_meter_data.csv"

# Load the data
data = pd.read_csv(input_csv_path)

# Split the data (80% train, 20% validation)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create directories for train and validation if they don't exist

# Save the datasets
train_data.to_csv("data/train/non_smart_meter_data_split_train.csv", index=False)
val_data.to_csv("data/val/non_smart_meter_data_split_val.csv", index=False)

print("Data has been successfully split and saved!")
