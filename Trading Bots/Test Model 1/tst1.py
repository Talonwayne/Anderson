
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Callable

# Custom StandardScaler
class CustomStandardScaler:
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        return self.fit(data).transform(data)

# Financial indicator calculations
def calculate_return_series(series: pd.Series) -> pd.Series:
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1

def calculate_log_return_series(series: pd.Series) -> pd.Series:
    shifted_series = series.shift(1, axis=0)
    return pd.Series(np.log(series / shifted_series))

def calculate_simple_moving_average(series: pd.Series, n: int=20) -> pd.Series:
    return series.rolling(n).mean()

def calculate_macd_oscillator(series: pd.Series, n1: int=5, n2: int=34) -> pd.Series:
    assert n1 < n2, 'n1 must be less than n2'
    return calculate_simple_moving_average(series, n1) - calculate_simple_moving_average(series, n2)

def calculate_bollinger_bands(series: pd.Series, n: int=20) -> pd.DataFrame:
    sma = calculate_simple_moving_average(series, n)
    stdev = series.rolling(n).std()
    return pd.DataFrame({'middle': sma, 'upper': sma + 2 * stdev, 'lower': sma - 2 * stdev})

def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    mfv = df['Volume'] * (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])
    return mfv

def calculate_money_flow_volume(df: pd.DataFrame, n: int=20) -> pd.Series:
    return calculate_money_flow_volume_series(df).rolling(n).sum()

def calculate_chaikin_money_flow(df: pd.DataFrame, n: int=20) -> pd.Series:
    return calculate_money_flow_volume(df, n) / df['Volume'].rolling(n).sum()

def calculate_targets(df: pd.DataFrame, days: int) -> pd.DataFrame:
    # Initialize a list to store the new column values
    new_col = []
    
    # Iterate over each row
    for i in range(len(df)):
        if i + days < len(df):
            # Get the 'Low' value of the current day
            current_low = df['Low'].iloc[i]
            
            # Get the 'Low' values of the next 'days' days
            next_lows = df['Low'].iloc[i+1:i+1+days]
            
            # Check if the current 'Low' is the lowest
            if current_low < next_lows.min():
                new_col.append(1)
            else:
                new_col.append(0)
        else:
            # If there are not enough future days to compare, assign 0
            new_col.append(0)
    
    # Append the new column to the DataFrame
    df['targets'] = new_col
    
    return df

# Load and preprocess the data
file_path = '/Users/talonwayneanderson/Desktop/Code/MachineLearning/Silver_data.csv'

# Debug: Check if the file path is correct
print(f"Loading data from {file_path}")

try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

days_to_buy = 14
data = calculate_targets(data, days_to_buy)
# Debug: Check the first few rows of the data
print(data.head())

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)
data.set_index('Date', inplace=True)
data.rename(columns={'Close/Last': 'Close'}, inplace=True)

# Calculate additional features
data['return'] = calculate_return_series(data['Close'])
data['log_return'] = calculate_log_return_series(data['Close'])
data['sma'] = calculate_simple_moving_average(data['Close'], 10)
data['macd'] = calculate_macd_oscillator(data['Close'], 5, 50)
bollinger_bands = calculate_bollinger_bands(data['Close'], 20)
data['bb_upper'] = bollinger_bands['upper']
data['bb_lower'] = bollinger_bands['lower']
data['cmf'] = calculate_chaikin_money_flow(data)
data.dropna(inplace=True)  # Drop rows with NaN values

# Debug: Check the first few rows of the processed data
print(data.head())

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'log_return', 'sma', 'macd', 'bb_upper', 'bb_lower', 'cmf']
X = data[features]
y = data['targets']

# Split into training and testing datasets
split_idx = int(len(data) * 0.8)
train_data, test_data = X[:split_idx], X[split_idx:]
train_target, test_target = y[:split_idx], y[split_idx:]

# Debug: Check the shapes of the datasets
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Create sequences for LSTM/RNN input
sequence_length = 30

class FinancialDataset(Dataset):
    def __init__(self, data, target, sequence_length):
        self.data = data
        self.target = target
        self.sequence_length = sequence_length
        self.target = self.target.reshape(-1, 1)  # Reshape target to (N, 1)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.target[idx:idx + self.sequence_length].reshape(-1, 1)  # Reshape to (sequence_length, 1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create DataLoaders
batch_size = 128
train_dataset = FinancialDataset(train_data.values, train_target.values, sequence_length)
test_dataset = FinancialDataset(test_data.values, test_target.values, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)








#SNN
"""
# Define the network
class FinancialNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        num_hidden1 = 256
        num_hidden2 = 128
        num_outputs = 1  # Single output for buy or not

        beta1 = 0.9
        beta2 = torch.rand((num_outputs), dtype=torch.float)
        beta3 = torch.rand((num_outputs), dtype=torch.float)

        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta2)
        self.fc3 = nn.Linear(num_hidden2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta3, learn_beta=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(features)
net = FinancialNet(input_dim).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = nn.BCELoss()

# Training loop parameters
num_epochs = 2
num_steps = 25

# Training loop

# Training loop

loss_hist = []
acc_hist = []

for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = net(data)
        spk_rec = spk_rec[-1]  # Use the output of the last time step

        loss_val = loss_fn(spk_rec, targets.float())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())

        if i % 25 == 0:
            net.eval()
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = (spk_rec.round() == targets).float().mean().item()
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

print(f"Trained decay rate of the first layer: {net.lif1.beta:.3f}\n")
print(f"Trained decay rates of the second layer: {net.lif2.beta}")
print(f"Trained decay rates of the third layer: {net.lif3.beta}")

def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)
            spk_rec = spk_rec[-1]  # Use the output of the last time step

            # Ensure the shape matches the target
            targets = targets.squeeze(-1)

            acc += (spk_rec.round() == targets).float().mean().item() * spk_rec.size(0)
            total += spk_rec.size(0)

    return acc / total

# Test set accuracy
print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps) * 100:.3f}%")
"""









"""# Training loop
loss_hist = []
acc_hist = []

for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = net(data)  # spk_rec shape: [sequence_length, batch_size, num_outputs]
        spk_rec = spk_rec[-1]   # Use the output of the last time step, shape: [batch_size, num_outputs]

        # Reshape targets to match spk_rec
        targets = targets[:, -1]  # shape: [batch_size]

        loss_val = loss_fn(spk_rec, targets.float())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())

        if i % 25 == 0:
            net.eval()
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = (spk_rec.round() == targets).float().mean().item()
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

print(f"Trained decay rate of the first layer: {net.lif1.beta:.3f}\n")
print(f"Trained decay rates of the second layer: {net.lif2.beta}")
print(f"Trained decay rates of the third layer: {net.lif3.beta}")

def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)
            spk_rec = spk_rec[-1]  # Use the output of the last time step

            # Reshape targets to match spk_rec
            targets = targets[:, -1]

            acc += (spk_rec.round() == targets).float().mean().item() * spk_rec.size(0)
            total += spk_rec.size(0)

    return acc / total

# Test set accuracy
print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps) * 100:.3f}%")
"""