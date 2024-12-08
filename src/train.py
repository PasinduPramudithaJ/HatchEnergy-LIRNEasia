import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load training and validation data
train_file_path = 'data/train/cleandata_train.csv'  # Replace with your actual training data path
val_file_path = 'data/val/clean_data_val.csv'      # Replace with your actual validation data path

train_data = pd.read_csv(train_file_path)
val_data = pd.read_csv(val_file_path)

# Define features and target
features = ['boiling_factor', 'NoOfHoursStay', 'no_of_household_members', 'Appliances', 'TotWatageAC']
target_consumption = 'consumption'  # Replace with the column name for consumption

# Extract features and target
X_train = train_data[features].values
y_train_consumption = train_data[target_consumption].values

X_val = val_data[features].values
y_val_consumption = val_data[target_consumption].values

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add a dimension for timesteps
y_train_c_tensor = torch.tensor(y_train_consumption, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
y_val_c_tensor = torch.tensor(y_val_consumption, dtype=torch.float32).unsqueeze(1)

# Define the RNN model
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


# Hyperparameters
input_size = len(features)
hidden_size = 64
learning_rate = 0.001
epochs = 1000
batch_size = 32

# Instantiate the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size)
criterion_consumption = nn.MSELoss()  # Loss for consumption prediction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader for batch processing
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_c_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# RMSE Calculation
def rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean())

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss_c = 0
    train_rmse = 0
    for batch in train_loader:
        X_batch, y_c_batch = batch

        # Forward pass
        pred_c = model(X_batch)

        # Compute loss
        loss_c = criterion_consumption(pred_c, y_c_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()

        # Accumulate loss and RMSE
        train_loss_c += loss_c.item()
        train_rmse += rmse(pred_c, y_c_batch).item()

    # Validation
    model.eval()
    with torch.no_grad():
        pred_c_val = model(X_val_tensor)
        val_loss_c = criterion_consumption(pred_c_val, y_val_c_tensor)
        val_rmse = rmse(pred_c_val, y_val_c_tensor).item()

    # Print the statistics for the current epoch
    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Consumption Loss: {train_loss_c / len(train_loader):.4f}, "
          f"Train RMSE: {train_rmse / len(train_loader):.4f}, "
          f"Val Consumption Loss: {val_loss_c:.4f}, "
          f"Val RMSE: {val_rmse:.4f}")

# Save the model
torch.save(model.state_dict(), 'consumption_rnn_model.pth')
print("Model saved as consumption_rnn_model.pth")

# Example: Making Predictions
new_data = np.array([[0.5, 4, 3, 200, 300]])  # Replace with new sample data
new_data_scaled = scaler.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).unsqueeze(1)

model.eval()
with torch.no_grad():
    consumption_pred = model(new_data_tensor)
    print(f"Predicted Consumption: {consumption_pred.item()}")
