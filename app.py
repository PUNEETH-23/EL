import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO(r'C:\Users\Puneeth Kumar\runs\detect\train10\weights\best.pt')  # Adjust the path as needed

# Streamlit UI
st.set_page_config(page_title="Rice Detection with YOLO", page_icon="🌾")
st.title("🌾 Rice Detection with YOLO")
st.write("Upload an image to perform object detection on rice grains.")

# Sidebar for better navigation and additional options
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform inference on the uploaded image with specified confidence threshold
    results = model(image, conf=confidence_threshold)

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
    st.subheader("Class Counts")
    for label, count in class_counts.items():
        st.write(f"**{label}**: {count}")

    # Visualize the results
    image_with_boxes = results[0].plot()  # This adds bounding boxes to the image
    image_with_boxes_pil = Image.fromarray(image_with_boxes)  # Convert to PIL Image format

    # Display the image with predictions in Streamlit
    st.image(image_with_boxes_pil, caption='Detected Objects', use_container_width=True)
else:
    st.write("Please upload an image file.")



