import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('./data/cleandata/merged_output.csv')

# Data Cleaning
# Drop rows where the target (Consumption) is missing
data = data.dropna(subset=['Consumption'])

# Fill missing values in predictors with median values
data['SumOfAppliances'] = data['SumOfAppliances'].fillna(data['SumOfAppliances'].median())
data['TotWatageAC'] = data['TotWatageAC'].fillna(data['TotWatageAC'].median())

# Select features and target
features = ['boiling_factor', 'NoOfHoursStay', 'no_of_household_members', 'SumOfAppliances', 'TotWatageAC']
target = 'Consumption'

X = data[features].values
y = data[target].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define Dataset class
class ConsumptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
train_dataset = ConsumptionDataset(X_train, y_train)
test_dataset = ConsumptionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use the last output of the sequence
        return out

# Model parameters
input_size = X_train.shape[1]
hidden_size = 64
num_layers = 2
output_size = 1

model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(1)  # Add sequence dimension
        y_batch = y_batch

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        predicted = outputs.round()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'rnn_model.pth')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.unsqueeze(1)  # Add sequence dimension
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

        # Accuracy calculation
        predicted = outputs.round()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%')
