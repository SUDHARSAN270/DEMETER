import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("D:\intelligent irrigation\data\data.csv")  # Replace with your actual file

# Encode 'crop' column (categorical to numerical)
label_encoder = LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['crop'])

# Define features and target
features = ['crop', 'moisture', 'temp']
target = 'pump'

X = df[features].values
y = df[target].values

# Normalize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class IrrigationNN(nn.Module):
    def __init__(self):
        super(IrrigationNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Input size = 3 (crop, moisture, temp)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()  # Binary classification (pump ON/OFF)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss, and optimizer
model = IrrigationNN()
criterion = nn.BCELoss()  # Binary Classification Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "irrigation_model.pth")
print("Model trained and saved!")

# Load trained model
model.load_state_dict(torch.load("irrigation_model.pth"))
model.eval()

# Make predictions on test data
with torch.no_grad():
    predictions = model(X_test)
    predictions_binary = (predictions > 0.5).float()  # Convert to 0 or 1

# Print actual vs predicted values
print("\nPredicted vs Actual Pump Status:")
for i in range(len(predictions)):
    print(f"Moisture: {X_test[i][1]:.2f}, Temp: {X_test[i][2]:.2f} | Predicted: {predictions_binary[i].item():.0f} | Actual: {y_test[i].item():.0f}")

# Calculate accuracy
accuracy = (predictions_binary.eq(y_test).sum() / float(y_test.shape[0])).item()
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

def predict_pump(crop, moisture, temp):
    # Encode crop type
    crop_encoded = label_encoder.transform([crop])[0]

    # Normalize moisture & temperature
    input_data = np.array([[crop_encoded, moisture, temp]])
    input_data = scaler.transform(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        pump_status = (prediction > 0.5).float().item()

    return "ON" if pump_status == 1 else "OFF"


joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Example sensor inputs
crop_type = "cotton"  # Example input (change as needed)
moisture_level = -0.50 
temperature = -0.90

# Predict pump status
pump_decision = predict_pump(crop_type, moisture_level, temperature)
print(f"Pump Status: {pump_decision}")




