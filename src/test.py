import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# Define the RNN model (same as during training)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_consumption = nn.Linear(hidden_size, 1)  # For consumption prediction

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Only use the final hidden state
        hidden = hidden[-1]
        consumption = self.fc_consumption(hidden)  # Predict consumption
        return consumption


# Load the trained model architecture and state_dict
input_size = 5  # Number of features used in the model
hidden_size = 64  # Same as used during training
model = RNNModel(input_size, hidden_size)  # Define the model architecture

# Load the model weights
model.load_state_dict(torch.load('consumption_rnn_model.pth'))
model.eval()

# Load the test data (replace with your actual file path)
test_file_path = 'data/test/test.csv'  # Replace with your test data file path
test_data = pd.read_csv(test_file_path)

# Define features (same as during training)
features = ['boiling_factor', 'NoOfHoursStay', 'no_of_household_members', 'Appliances', 'TotWatageAC']

# Extract features from the test data
X_test = test_data[features].values

# Normalize features using the same scaler as during training
scaler = MinMaxScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Convert to PyTorch tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)  # Add a dimension for timesteps

# Predict consumption using the trained model
with torch.no_grad():
    consumption_pred = model(X_test_tensor)

# Convert predictions to numpy
consumption_pred_np = consumption_pred.numpy()

# Define a threshold for efficient vs inefficient consumption (e.g., 100 kWh)
threshold = 100  # This value is an example, adjust based on your problem or data

# Classify as efficient or not based on the threshold
efficiency_pred = ["Efficient" if consumption <= threshold else "Not Efficient" for consumption in consumption_pred_np]

# Save predictions and efficiency status to a CSV file for review
predictions_df = pd.DataFrame({
    'Predicted_Consumption': consumption_pred_np.flatten(),
    'Efficiency_Status': efficiency_pred
})
predictions_df.to_csv('predictions_with_efficiency.csv', index=False)
print("Predictions with efficiency status saved to 'predictions_with_efficiency.csv'.")

# Print out some example predictions with efficiency status
for i in range(min(5, len(consumption_pred_np))):  # Show up to 5 predictions
    print(f"Predicted Consumption for sample {i+1}: {consumption_pred_np[i][0]}, Efficiency: {efficiency_pred[i]}")
