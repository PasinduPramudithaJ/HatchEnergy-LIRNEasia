import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the RNN model class
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # Initialize hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use the last output of the sequence
        return out

# Model parameters
input_size = 5  # Number of input features (e.g., boiling_factor, NoOfHoursStay, etc.)
hidden_size = 64
num_layers = 2
output_size = 1

# Load the trained model
model = RNNModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('rnn_model.pth'))  # Load model weights from the saved file
model.eval()  # Set the model to evaluation mode

# Load and preprocess the input data
csv_path = './data/test/test.csv'  # Path to your CSV file
data = pd.read_csv(csv_path)

# Handle missing values by filling with median (optional)
data['SumOfAppliances'] = data['SumOfAppliances'].fillna(data['SumOfAppliances'].median())
data['TotWatageAC'] = data['TotWatageAC'].fillna(data['TotWatageAC'].median())

# Select features to be used as input to the model
features = ['boiling_factor', 'NoOfHoursStay', 'no_of_household_members', 'SumOfAppliances', 'TotWatageAC']
X = data[features].values

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert the features to a PyTorch tensor and add a sequence dimension
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension (batch_size, seq_len, input_size)

# Make predictions using the trained model
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X_tensor).squeeze(1).numpy()  # Get the predictions for all rows

# Add the predictions as a new column in the DataFrame
data['Predicted_Consumption'] = predictions

# Define the deviation threshold (based on your domain knowledge, adjust as needed)
deviation_threshold = 10  # You can adjust this threshold

# Calculate the absolute deviation between Actual Consumption and Predicted Consumption
data['Deviation'] = abs(data['Consumption'] - data['Predicted_Consumption'])

# Classify as 'Efficient' or 'Inefficient' based on the deviation
data['Efficiency'] = data['Deviation'].apply(lambda x: 'Efficient' if x <= deviation_threshold else 'Inefficient')

# Print the results: Actual Consumption, Predicted Consumption, Deviation, Efficiency
print(data[['Consumption', 'Predicted_Consumption', 'Deviation', 'Efficiency']])

# Optionally, save the results to a new CSV file
data.to_csv('predicted_consumption_results.csv', index=False)
print("Predictions saved to 'predicted_consumption_results.csv'.")
