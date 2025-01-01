from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO(r'C:\Users\Puneeth Kumar\runs\detect\train10\weights\best.pt')  # Adjust the path as needed

# Streamlit UI for file upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Perform inference on the uploaded image
    results = model(image)
    
    # Extract class names and confidence scores
    class_names = model.names  # Retrieve class names from the model

    # Initialize a dictionary to count occurrences of each class
    class_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])  # Get the class ID
        label = class_names[class_id]  # Map class ID to label

        # Count occurrences of each class
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Display the count of each class
    st.write("Class Counts:")
    for label, count in class_counts.items():
        st.write(f"{label}: {count}")

    # Optionally, visualize the results
    # Convert the results to a format that can be displayed using Streamlit
    image_with_boxes = results[0].plot()  # This adds bounding boxes to the image
    image_with_boxes_pil = Image.fromarray(image_with_boxes)  # Convert to PIL Image format

    # Display the image with predictions in Streamlit
    st.image(image_with_boxes_pil, caption='Detected Objects', use_container_width=True)
else:
    st.write("Please upload an image file.")
