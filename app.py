import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="BedBug | Edge Vision Inspector",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a dark mode/clean dashboard feel
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1e1e1e;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #2b2b2b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BedBug | Edge Vision Inspector 🐛")
st.markdown("A lightweight, edge-capable Computer Vision application for object detection.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Loads the YOLOv8-nano model. Cached to prevent reloading on every interaction."""
    # Using the nano model for edge optimization
    model = YOLO("yolov8n.pt")
    return model

try:
    with st.spinner("Loading AI Model..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Get model classes
class_names = model.names

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")

# Image Source Selection
image_source = st.sidebar.radio(
    "Image Source Selection",
    ["Upload Image", "Webcam Capture"]
)

st.sidebar.subheader("Model Parameters")
# Confidence Threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for a detection to be considered valid."
)

# IoU Overlap Threshold
iou_threshold = st.sidebar.slider(
    "IoU Overlap Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for Non-Maximum Suppression (NMS)."
)

# Class Filter
st.sidebar.subheader("Class Filter")
# Provide a multiselect for classes. Default to all classes if none selected later, but pre-select a few for demo.
all_classes_list = list(class_names.values())
selected_classes = st.sidebar.multiselect(
    "Select classes to detect:",
    options=all_classes_list,
    default=all_classes_list[:5], # Default to first 5 just so it's not empty, or user can clear
    help="Only show detections for these selected classes."
)

# Map selected class names back to their integer IDs for the model
selected_class_ids = [k for k, v in class_names.items() if v in selected_classes]

# Diagnostic Stats Toggle
show_metrics = st.sidebar.checkbox("Show Performance Metrics", value=True)


# --- Main Application Logic ---

image = None

# 1. Image Acquisition
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded image: {e}")
elif image_source == "Webcam Capture":
    camera_image = st.camera_input("Capture an image from your webcam")
    if camera_image is not None:
         try:
             image = Image.open(camera_image)
         except Exception as e:
             st.error(f"Error reading webcam image: {e}")

# 2. Inference & Display
if image is not None:
    # Convert PIL Image to RGB (OpenCV format uses BGR, but YOLO expects RGB PIL or numpy array)
    image_np = np.array(image.convert("RGB"))

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    with st.spinner("Running Inference..."):
        try:
            start_time = time.time()
            # Run inference
            # If selected_class_ids is empty, model detects all classes by default, 
            # but we can enforce the filter by passing classes=selected_class_ids if not empty
            kwargs = {"conf": conf_threshold, "iou": iou_threshold}
            if selected_class_ids:
                kwargs["classes"] = selected_class_ids
                
            results = model(image_np, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            result = results[0]
            
            # Plot results on image
            # result.plot() returns a BGR numpy array
            processed_image_bgr = result.plot()
            processed_image_rgb = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")
            result = None

    if result is not None:
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image_rgb, use_container_width=True)
            
        # 3. Metrics & Data Extraction
        boxes = result.boxes
        num_detections = len(boxes)
        
        # High vs Low confidence (threshold at 0.50 for this breakdown)
        confidences = boxes.conf.cpu().numpy()
        high_conf_count = int(np.sum(confidences >= 0.50))
        low_conf_count = int(np.sum(confidences < 0.50))

        if show_metrics:
            st.markdown("---")
            st.subheader("Diagnostic Stats")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Targets Detected", num_detections)
            m2.metric("Inference Latency (ms)", f"{latency_ms:.1f}")
            m3.metric("High Confidence (≥0.5)", high_conf_count)
            m4.metric("Low Confidence (<0.5)", low_conf_count)

        # 4. Data Export
        if num_detections > 0:
            st.markdown("---")
            with st.expander("Detection Results Data"):
                # Extract data for dataframe
                class_ids = boxes.cls.cpu().numpy().astype(int)
                coords = boxes.xyxy.cpu().numpy() # [xmin, ymin, xmax, ymax]
                
                data = {
                    "Class": [class_names[c_id] for c_id in class_ids],
                    "Confidence Score": np.round(confidences, 3),
                    "xmin": np.round(coords[:, 0], 2),
                    "ymin": np.round(coords[:, 1], 2),
                    "xmax": np.round(coords[:, 2], 2),
                    "ymax": np.round(coords[:, 3], 2),
                }
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # CSV Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="detection_results.csv",
                    mime="text/csv",
                )
        else:
            st.info("No objects detected with the current confidence threshold and class filters.")
else:
    st.info("Please upload an image or capture from webcam to start inspection.")
