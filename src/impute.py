import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
file_path = './data/cleandata/merged_output.csv'  # Path to your CSV file
data = pd.read_csv(file_path)

# Define the features to impute
features_to_impute = ['Consumption', 'boiling_factor', 'NoOfHoursStay', 'no_of_household_members', 'SumOfAppliances', 'TotWatageAC']

# Check for missing values before imputation
print("Missing values before imputation:")
print(data[features_to_impute].isnull().sum())

# Create an instance of KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)  # Adjust the number of neighbors if necessary

# Apply KNN imputation
data[features_to_impute] = knn_imputer.fit_transform(data[features_to_impute])

# Check for missing values after imputation
print("Missing values after imputation:")
print(data[features_to_impute].isnull().sum())

# Save the imputed dataset to a new CSV file
output_file_path = './data/cleandata/merged_output.csv'
data.to_csv(output_file_path, index=False)

print(f"Imputed dataset saved as {output_file_path}")
