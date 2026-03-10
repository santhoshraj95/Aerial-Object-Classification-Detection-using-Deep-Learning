import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import os

# Set page config
st.set_page_config(
    page_title="Bird vs Drone Detector",
    page_icon="🛸",
    layout="centered"
)

# Simple CSS for clean UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .bird-prediction {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .drone-prediction {
        background-color: #ffe8e8;
        border: 2px solid #f44336;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# PyTorch Model
class BirdDroneClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BirdDroneClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_classification_model():
    """Load PyTorch classification model"""
    try:
        model = BirdDroneClassifier(num_classes=2)
        model_path = "bird_drone_model.pth"  # Simplified path for deployment
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Classification model error: {e}")
        return None

@st.cache_resource
def load_detection_model():
    """Load YOLOv8 detection model"""
    try:
        detector = YOLO('yolov8n.pt')
        return detector
    except Exception as e:
        st.warning(f"Detection model not available: {e}")
        return None

def preprocess_image(image, size=(224, 224)):
    """Preprocess image for classification"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_rgb = image.convert('RGB')
    return transform(image_rgb).unsqueeze(0)

def classify_image(model, image):
    """Run classification"""
    if model is None:
        return "Model not loaded", 0, []
    
    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)
        
        classes = ["Bird", "Drone"]
        predicted_class = classes[predicted.item()]
        confidence = conf.item()
        prob_values = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, prob_values
    except Exception as e:
        return f"Error: {e}", 0, []

def detect_objects(model, image):
    """Run object detection"""
    if model is None:
        return [], [], [], {}
    
    try:
        image_np = np.array(image)
        results = model(image_np, conf=0.25)
        result = results[0]
        
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        class_names = result.names
        
        return boxes, classes, confidences, class_names
    except Exception as e:
        return [], [], [], {}

def draw_bounding_boxes(image, boxes, classes, confidences, class_names):
    """Draw bounding boxes on image"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(cls)]
        
        # Choose color based on object type
        if 'bird' in class_name.lower():
            color = "green"
        elif 'airplane' in class_name.lower():
            color = "red"
        else:
            color = "blue"
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        try:
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(label) * 10
            text_height = 20
        
        # Draw label background
        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 3), label, fill="white", font=font)
    
    return draw_image

def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🛸 Bird vs Drone Detector")
    st.write("Upload an image to classify between birds and drones")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models
    classification_model = load_classification_model()
    detection_model = load_detection_model()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a bird or drone"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Classification
        st.subheader("🔍 Classification Result")
        
        if classification_model:
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = classify_image(classification_model, image)
            
            if not predicted_class.startswith("Error"):
                # Display prediction
                prediction_class = "bird-prediction" if predicted_class == "Bird" else "drone-prediction"
                emoji = "🐦" if predicted_class == "Bird" else "🚁"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2>{emoji} {predicted_class}</h2>
                    <h3>Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bars
                st.subheader("📊 Confidence Scores")
                col1, col2 = st.columns(2)
                
                with col1:
                    bird_prob = probabilities[0]
                    st.metric("Bird", f"{bird_prob:.2%}")
                    st.progress(float(bird_prob))
                
                with col2:
                    drone_prob = probabilities[1]
                    st.metric("Drone", f"{drone_prob:.2%}")
                    st.progress(float(drone_prob))
            else:
                st.error(predicted_class)
        else:
            st.error("Classification model not available")
        
        # Object Detection (Optional)
        st.subheader("🎯 Object Detection")
        
        if detection_model:
            with st.spinner("Detecting objects..."):
                boxes, classes, confidences, class_names = detect_objects(detection_model, image)
            
            if len(boxes) > 0:
                st.success(f"Found {len(boxes)} objects")
                
                # Draw bounding boxes
                detection_image = draw_bounding_boxes(image, boxes, classes, confidences, class_names)
                st.image(detection_image, caption="Detection Results", use_container_width=True)
                
                # Show detection details
                st.write("**Detected Objects:**")
                for i, (cls, conf) in enumerate(zip(classes, confidences)):
                    class_name = class_names[int(cls)]
                    st.write(f"- {class_name} ({conf:.2%} confidence)")
            else:
                st.info("No objects detected")
        else:
            st.info("Object detection not available")
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload an image to get started")
        
        # Sample layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**What this app does:**")
            st.write("• Classifies images as Bird or Drone")
            st.write("• Shows confidence scores")
            st.write("• Detects objects with bounding boxes")
            st.write("• Uses PyTorch + YOLOv8 models")
        
        with col2:
            st.write("**Supported formats:**")
            st.write("• JPG, JPEG, PNG")
            st.write("**Best results:**")
            st.write("• Clear, well-lit images")
            st.write("• Centered subjects")


import base64

def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        /* Optional: Make content background transparent */
        [data-testid="stMarkdownContainer"] p,
        .stButton button,
        .stAlert,
        .css-ffhzg2 {{
            background: rgba(0,0,0,0.4);
            color: white !important;
            border-radius: 10px;
            padding: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the background function
add_bg_from_local(r"C:\Users\lenovo\Downloads\freepik__the-style-is-candid-image-photography-with-natural__22181.png")


if __name__ == "__main__":
    main()