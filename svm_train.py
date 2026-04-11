import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

# ==========================================
# Path Configuration (Localized)
# ==========================================
dataset_path = "svm_dataset"
model_save_path = "svm_color_model.pkl"

def extract_features(image_path):
    # Support for non-ASCII paths (like some system names)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def train_model():
    print("=== Starting SVM Color Model Training ===")
    features, labels = [], []
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found.")
        return

    for color_name in os.listdir(dataset_path):
        color_dir = os.path.join(dataset_path, color_name)
        if not os.path.isdir(color_dir): continue
            
        print(f"Reading class: {color_name} ...")
        for img_name in os.listdir(color_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(color_dir, img_name)
            feature = extract_features(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(color_name)
                
    if not features:
        print("Warning: No valid image data found!")
        return

    print("Data loaded. Training SVM model...")
    # Using linear kernel with probability enabled for confidence output
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(features, labels)
    
    joblib.dump(svm_model, model_save_path)
    print(f"Success! Model saved to: {model_save_path}")

if __name__ == "__main__":
    train_model()
