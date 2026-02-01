# 🛰️ Aerial Object Classification & Detection

## 📌 Project Overview

This project focuses on building a **Deep Learning–based Computer Vision solution** to classify aerial images into **Bird** or **Drone**, and optionally perform **object detection** to localize these objects in real-world aerial scenes.

The system is designed for **aerial surveillance, wildlife monitoring, airport safety, and security & defense applications**, where distinguishing between birds and drones is critical.

The solution includes:

* Image Classification using **Custom CNN** and **Transfer Learning models**
* Optional **Object Detection using YOLOv8**
* **Streamlit web application** for interactive deployment

---

## 🎯 Problem Statement

To develop a robust AI system that:

* Accurately classifies aerial images as **Bird** or **Drone**
* Optionally detects and localizes birds/drones in images using bounding boxes
* Provides a user-friendly interface for real-time prediction

---

## 🧠 Skills & Technologies

* Deep Learning
* Computer Vision
* Image Classification & Object Detection
* Python
* TensorFlow / Keras or PyTorch
* Data Preprocessing & Augmentation
* Transfer Learning (ResNet50, MobileNet, EfficientNetB0)
* YOLOv8 (Optional)
* Model Evaluation & Comparison
* Streamlit Deployment

---

## 🌍 Domain Applications

* **Wildlife Protection** – Detect birds near wind farms or airports
* **Security & Defense** – Identify unauthorized drones in restricted airspace
* **Airport Bird-Strike Prevention** – Monitor runway bird activity
* **Environmental Research** – Track bird populations from aerial data

---

## 📊 Datasets

### 📌 Classification Dataset

**Source:** `classification_dataset`

**Task:** Binary Image Classification (Bird / Drone)

**Format:** RGB Images (.jpg)

**Dataset Structure:**

```
classification_dataset/
│
├── train/
│   ├── bird   (1414 images)
│   └── drone  (1248 images)
│
├── valid/
│   ├── bird   (217 images)
│   └── drone  (225 images)
│
└── test/
    ├── bird   (121 images)
    └── drone  (94 images)
```

---

### 📌 Object Detection Dataset (YOLOv8 Format)

**Source:** `object_detection_Dataset`

* Total Images: **3319**
* Annotation Format: YOLOv8 (`.txt`)

**Annotation Structure:**

```
<class_id> <x_center> <y_center> <width> <height>
```

**Data Split:**

* Train: 2662 images
* Validation: 442 images
* Test: 215 images

---

## 🔄 Project Workflow

### 1️⃣ Dataset Understanding

* Inspect folder structure
* Count images per class
* Identify class imbalance
* Visualize sample images

### 2️⃣ Data Preprocessing

* Resize images to **224 × 224**
* Normalize pixel values to **[0, 1]**
* Apply model-specific preprocessing:

  * TensorFlow: `preprocess_input()`
  * PyTorch: ImageNet mean & std normalization

### 3️⃣ Data Augmentation

* Rotation
* Horizontal/Vertical flipping
* Zoom
* Brightness adjustment
* Random cropping

### 4️⃣ Model Building (Classification)

* **Custom CNN**:

  * Convolution layers
  * MaxPooling
  * Batch Normalization
  * Dropout
  * Fully Connected Layers

* **Transfer Learning**:

  * ResNet50
  * MobileNet
  * EfficientNetB0
  * Fine-tuning for improved accuracy

### 5️⃣ Model Training

* Train Custom CNN & Transfer Learning models
* Use:

  * EarlyStopping
  * ModelCheckpoint
* Metrics tracked:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

### 6️⃣ Model Evaluation

* Confusion Matrix
* Classification Report
* Accuracy & Loss Curves

### 7️⃣ Model Comparison

* Compare:

  * Accuracy
  * Training time
  * Generalization performance
* Select best model for deployment

---

## 🚀 Optional: Object Detection with YOLOv8

### Steps:

1. Install YOLOv8
2. Prepare dataset in YOLO format (completed)
3. Create `data.yaml`
4. Train YOLOv8 model
5. Validate trained model
6. Run inference on test/new images

---

## 🌐 Streamlit Deployment

### Features:

* Image upload interface
* Display prediction: **Bird / Drone**
* Confidence score visualization
* Optional YOLOv8 bounding box detection

---

## 📦 Project Deliverables

* ✅ Trained Models:

  * Custom CNN
  * Transfer Learning Model
  * YOLOv8 (Optional)
* ✅ Streamlit Web Application
* ✅ Preprocessing, Training & Evaluation Scripts
* ✅ Model Comparison Report
* ✅ GitHub Repository with Documentation
* ✅ Clean, modular & well-commented code

---

## 🛠 Technical Tags

`Computer Vision` `Deep Learning` `CNN` `Image Classification` `Object Detection` `YOLOv8` `Transfer Learning` `Data Augmentation` `Model Evaluation` `Streamlit` `Aerial Surveillance AI`




