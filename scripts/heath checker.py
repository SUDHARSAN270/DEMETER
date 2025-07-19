import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans
from pathlib import Path


train_dataset = "D:\intelligent irrigation\data\ytrain"
test_dataset=  "D:\intelligent irrigation\data\ytest"


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image_folder = Path("D:\intelligent irrigation\data\ytrain")
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


train_dataset_path = "D:\intelligent irrigation\data\ytrain"
test_dataset_path = "D:\intelligent irrigation\data\ytest"


train_dataset = UnlabeledImageDataset(train_dataset_path, transform=transform)
test_dataset = UnlabeledImageDataset(test_dataset_path, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


for images in train_loader:
    print(f"Loaded batch with {images.shape}")
    break

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

train_dataset_path = "D:\intelligent irrigation\data\ytrain"
dataset = UnlabeledImageDataset(train_dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


features = []
image_paths = []

with torch.no_grad():
    for images, paths in dataloader:
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), -1)
        features.append(outputs.cpu().numpy())
        image_paths.extend(paths)

features = np.vstack(features)

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(features)


image_labels = {path: label for path, label in zip(image_paths, labels)}

print("Generated labels:", image_labels)

class LabeledImageDataset(Dataset):
    def __init__(self, image_folder, image_labels, transform=None):
        self.image_folder = image_folder
        self.image_labels = image_labels
        self.transform = transform
        self.image_files = list(image_labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.image_labels[img_path]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = LabeledImageDataset(train_dataset_path, image_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class CropHealthCNN(nn.Module):
    def __init__(self):
        super(CropHealthCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CropHealthCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "crop_health_model.pth")
print("Crop health classification model trained and saved!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropHealthCNN().to(device)
model.load_state_dict(torch.load("crop_health_model.pth", map_location=device))
model.eval()



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)


    class_labels = {0: "Healthy", 1: "Diseased"}
    prediction = class_labels[predicted.item()]

    print(f"Prediction: {prediction}")
    return prediction


image_path = "D:\intelligent irrigation\data\ytest\yvalid\7c691a49-8a56-49d2-856f-6c20c34e257b___RS_Erly-B-9586_JPG_jpg.rf.40b1682256692ed6ac29074b61fcef3c.jpg"
predict_image(image_path)

torch.save(model.state_dict(), "crop_health_model.pth")



