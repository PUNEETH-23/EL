import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO(r'C:\Users\Puneeth Kumar\runs\detect\train\weights\best.pt')  # Adjust the path as needed

# Streamlit UI Configuration
st.set_page_config(page_title="Rice Detection with YOLO", page_icon="🌾", layout="wide")
st.title("🌾 Rice Detection with YOLO")
st.write("Upload an image to perform object detection on rice grains.")

# Sidebar for options and navigation
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, help="Adjust the confidence threshold for object detection."
)

# File uploader for image upload
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

    # Calculate the percentage of whole grains
    whole = class_counts.get(class_names[0], 0)
    broken = class_counts.get(class_names[1], 0)
    try:
        percent = round((whole / (whole + broken)) * 100, 2)
    except ZeroDivisionError:
        percent = 0
    st.write(f"**Percentage : {percent}%**")

    # Visualize the results without labels
    image_with_boxes = np.array(image)  # Start with the original image
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box
        confidence = box.conf[0]  # Get the confidence score
        if confidence >= confidence_threshold:
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box in green

    # Convert to PIL Image format for display
    image_with_boxes_pil = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))

    # Display the image with predictions in Streamlit
    st.image(image_with_boxes_pil, caption='Detected Objects', use_container_width=True)
else:
    st.write("Please upload an image file.")

# Footer for additional information or links
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses the YOLO model to detect and classify rice grains into different categories. "
    "Adjust the confidence threshold to improve detection accuracy. Upload an image to see the results."
)
