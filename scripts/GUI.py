import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import serial  
import threading
import time
import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2


# === LOAD MODELS ===

# Irrigation model
class IrrigationNN(nn.Module):
    def __init__(self):
        super(IrrigationNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

irrigation_model = IrrigationNN()
irrigation_model.load_state_dict(torch.load("irrigation_model.pth", map_location="cpu"))
irrigation_model.eval()

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Health checker model
class CropHealthCNN(nn.Module):
    def __init__(self):
        super(CropHealthCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

health_model = CropHealthCNN()
health_model.load_state_dict(torch.load("crop_health_model.pth", map_location="cpu"))
health_model.eval()

# Transform for image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === SERIAL SETUP ===
SERIAL_PORT = "COM3"  # CHANGE THIS
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
except Exception as e:
    print("⚠️ Serial connection failed:", e)
    arduino = None

# === GUI ===
root = tk.Tk()
root.title("DEMETER")
root.geometry("600x600")
root.configure(bg="#111")
navbar_frame = tk.Frame(root, bg="#1a1a1a", height=50)
navbar_frame.pack(fill='x', side='top')

pages = {}

def show_page(page_name):
    # Hide all pages
    for page in pages.values():
        page.pack_forget()
    # Show selected page
    pages[page_name].pack(fill="both", expand=True)


content_frame = tk.Frame(root, bg="#262626")
content_frame.pack(fill="both", expand=True)
# Create separate frames for each page
def create_page(name):
    frame = tk.Frame(content_frame, bg="#262626")
    label = tk.Label(frame, text=f"{name} PAGE", font=("Orbitron", 24), fg="white", bg="#262626")
    label.pack(pady=50)
    return frame

nav_items = ["DASH BOARD", "MODELS", "REQUESTS", "SETTINGS"]

info_label = None
result_label = None

for item in nav_items:
    pages[item] = create_page(item)

    # ➕ Custom content for each page
    if item == "DASH BOARD":
        tk.Label(pages[item], text="Smart Irrigation System Dashboard", font=("Consolas", 14, "bold"),
                 fg="cyan", bg="#262626").pack(pady=10)

        dashboard_info = """🌱 Welcome to the Smart Irrigation System.
        
This AI-powered system helps farmers and gardeners optimize water usage using:

✔️ AI prediction for irrigation
✔️ Real-time sensor input (crop, moisture, temperature)
✔️ Automated pump control via Arduino
        
Stay efficient. Save water. Grow smarter."""

        tk.Label(pages[item], text=dashboard_info, font=("Consolas", 10), fg="white", bg="#262626",
                 justify="left", wraplength=500).pack(pady=5)

    elif item == "MODELS":
        tk.Label(pages[item], text="AIble", font=("Consolas", 14, "bold"),
                 fg="lightgreen", bg="#26262 Models Availa6").pack(pady=10)

        model1_text = """🔹 IrrigationNN (Neural Network)
- Input: Crop Type, Moisture Level, Temperature
- Layers: [3 ➝ 16 ➝ 8 ➝ 1]
- Activation: ReLU + Sigmoid
- Output: 1 = Pump ON, 0 = OFF"""

        model2_text = """🔸 CropHealthCNN (Convolutional Neural Network)
- Input: RGB Image (e.g., Leaf)
- Conv Layers: 3 ➝ 32 ➝ 64
- Fully Connected: [64×32×32 ➝ 128 ➝ 2]
- Output: 0 = Healthy, 1 = Diseased"""

        tk.Label(pages[item], text=model1_text, font=("Consolas", 10), fg="cyan", bg="#262626",
                 justify="left", wraplength=500).pack(pady=5)
        tk.Label(pages[item], text=model2_text, font=("Consolas", 10), fg="orange", bg="#262626",
                 justify="left", wraplength=500).pack(pady=5)

    elif item == "REQUESTS":
        if arduino and arduino.in_waiting > 0:
            communication_status = " ✅ connection established"
            info_color ="lightgreen"
        else:
            communication_status = "⚠️ Waiting for Arduino connection..."
            info_color ="red"

        
        tk.Label(pages[item], text="Real-time Arduino Communication", font=("Consolas", 14, "bold"),
                 fg="orange", bg="#262626").pack(pady=10)

        info_label = tk.Label(pages[item], text=communication_status, font=("Consolas", 10),
                          fg=info_color, bg="#262626")
        info_label.pack(pady=5)

        result_label = tk.Label(pages[item], text="🚀 System Ready", font=("Consolas", 12, "bold"),
                            fg="lightgreen", bg="#262626")
        result_label.pack(pady=10)

        # Function to update Arduino status in real-time
        def update_arduino_status():
            if arduino and arduino.in_waiting > 0:
                try:
                  line = arduino.readline().decode().strip()
                  if line:
                        parts = line.split(",")
                        if len(parts) == 3:
                            crop, moisture, temp = parts
                            info_label.config(text=f"🌾 Crop: {crop} | 💧 Moisture: {moisture}% | 🌡️ Temp: {temp}°C",
                                          fg="cyan")

                            status = predict_irrigation(crop, moisture, temp)
                            if status == 1:
                                 result = "🚿 Motor Status: ON 🔴"
                                 arduino.write(b'1')
                            elif status == 0:
                                 result = "✅ Parameters OK. Motor OFF 🟢"
                                 arduino.write(b'0')
                            else:
                                 result = "⚠️ Invalid data received."

                            result_label.config(text=result)

                except Exception as e:
                             result_label.config(text="⚠️ Error in data processing.")
                             print("Error reading serial:", e)

            pages["REQUESTS"].after(1000, update_arduino_status)  # Refresh every second

    # Start updating when this page loads
            pages["REQUESTS"].after(1000, update_arduino_status)

    elif item == "SETTINGS":
        tk.Label(pages[item], text="Adjust your system preferences here.",
                 font=("Consolas", 12), fg="white", bg="#333333", wraplength=500).pack(pady=10)


    # Create navbar button
    btn = tk.Label(navbar_frame, text=item, font=("Orbitron", 12, "bold"),
                   fg="white", bg="#1a1a1a", padx=20, pady=10)
    btn.pack(side="left", padx=5)

    # Bind hover effect
    btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#333333"))
    btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#1a1a1a"))

    # Bind click to switch page
    btn.bind("<Button-1>", lambda e, name=item: show_page(name))

# Show default page
show_page("DASH BOARD")

info_label = tk.Label(root, text="Waiting for data from Arduino...", fg="cyan", bg="#111", font=("Consolas", 12))
info_label.pack(pady=20)

result_label = tk.Label(root, text="", fg="white", bg="#111", font=("Consolas", 12))
result_label.pack()

# === Health Check ===
image_path_var = tk.StringVar()



def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    image_path_var.set(file_path)

def predict_health():
    path = image_path_var.get()
    if not path:
        result_label.config(text="Please upload an image for health check.")
        return

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = health_model(image)
        _, pred = torch.max(out, 1)
        health = "Healthy ✅" if pred.item() == 0 else "Diseased ⚠️"
        if pred.item() == 1:
            result_label.config(text="🌿 Diseased! Sending alert to Arduino 🚨", fg="red")
            if arduino:
             arduino.write(b'2')
        else:
            result_label.config(text="🌿 Healthy crop ✅", fg="lightgreen")
            if arduino:
                arduino.write(b'3')

    result_label.config(text=f"🧬 Crop Health: {health}")

tk.Button(root, text="Upload Leaf Image", command=browse_image).pack(pady=5)
tk.Button(root, text="Check Crop Health", command=predict_health).pack(pady=5)

# === Irrigation Prediction Function ===
def predict_irrigation(crop, moisture, temp):
    try:
        crop_encoded = label_encoder.transform([crop])[0]
        input_data = scaler.transform([[crop_encoded, float(moisture), float(temp)]])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            out = irrigation_model(input_tensor)
            result = (out > 0.5).float().item()
        return int(result)
    except:
        return -1  # invalid

# === Read from Arduino in background ===
def read_serial():
    while True:
        if arduino and arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode().strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 3:
                        crop, moisture, temp = parts
                        info_label.config(text=f"🌾 Crop: {crop} | 💧 Moisture: {moisture} | 🌡️ Temp: {temp}")
                        status = predict_irrigation(crop, moisture, temp)
                        if status == 1:
                            result = "🚿 Motor Status: ON 🔴"
                            arduino.write(b'1')
                        elif status == 0:
                            result = "✅ Parameters OK. Motor OFF 🟢"
                            arduino.write(b'0')
                        else:
                            result = "⚠️ Invalid data received."
                        result_label.config(text=result)
            except Exception as e:
                print("Error reading serial:", e)
        time.sleep(1)

threading.Thread(target=read_serial, daemon=True).start()

root.mainloop()
