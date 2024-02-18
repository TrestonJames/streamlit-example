import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from yolov5 import YOLOv5

# Load the YOLO model
model = YOLOv5("yolov5s", classes=2)  # "yolov5s" is the smallest model. Adjust model size as needed.

def detect_vehicles(image):
    # Convert the PIL image to a Torch tensor
    image = F.to_tensor(image).unsqueeze(0)
    # Perform inference
    results = model(image)
    # Filter detections for vehicles (cars, trucks, etc.)
    results = results.pandas().xyxy[0]  # Results as a DataFrame
    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']  # Define vehicle classes based on your YOLO model
    vehicles = results[results['name'].isin(vehicle_classes)]
    return vehicles

def main():
    st.title("Vehicle Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting vehicles...")

        vehicles = detect_vehicles(image)
        vehicle_count = len(vehicles)

        st.write(f"Detected {vehicle_count} vehicles in the image.")

if __name__ == "__main__":
    main()
