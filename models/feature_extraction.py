import os
import numpy as np
import cv2
import tensorflow as tf
import albumentations as A
from tensorflow.keras.models import load_model, Model
from sklearn.decomposition import PCA
from tqdm import tqdm

# Disable GPU explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Trained U-Net Model
unet_model_path = "C:/Users/nniha/flask_env/Scripts/project/models/Unet.keras"
if not os.path.exists(unet_model_path):
    raise FileNotFoundError(f"❌ U-Net model not found at {unet_model_path}")

unet_model = load_model(unet_model_path, compile=False)

# Extract Feature Layer (Update this if layer name differs)
try:
    feature_extractor = Model(inputs=unet_model.input, outputs=unet_model.get_layer("conv5_block1_out").output)
except:
    raise ValueError("❌ Layer 'conv5_block1_out' not found. Check model structure.")

# Define Augmentation Pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.1),
])

def extract_features(img_path):
    """Extract feature vectors from an image using the U-Net model."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Skipped (Unreadable): {img_path}")
        return None  # Skip unreadable images
    
    img = cv2.resize(img, (128, 128)) / 255.0  # Resize and normalize
    augmented = augmentation(image=(img * 255).astype(np.uint8))['image'] / 255.0  # Apply augmentation
    img = np.expand_dims(augmented, axis=0)  # Add batch dimension
    
    # Extract features
    features = feature_extractor(img, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Flatten()(features)
    
    return np.squeeze(features)  # Converts (1, 2048) to (2048,)

# Load Dataset Paths
dataset_path = "C:/Users/nniha/flask_env/Scripts/project/static/Images/"
output_feature_path = "C:/Users/nniha/flask_env/Scripts/project/models/features.npy"
output_pca_feature_path = "C:/Users/nniha/flask_env/Scripts/project/models/features_pca.npy"
output_log_path = "C:/Users/nniha/flask_env/Scripts/project/models/missing_images.txt"

features = []
processed_images = []
skipped_images = []

# Process each category
categories = ["non_deforested", "deforested", "clear_cutting", "edge_deforested", "fragmented", "selective_logging"]
print("🔄 Extracting features from dataset...")
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder {folder_path} not found.")
        continue

    image_files = sorted(os.listdir(folder_path))  # Ensure consistent ordering
    for img_name in tqdm(image_files, desc=f"Processing {category}"):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, img_name)
            feature_vector = extract_features(img_path)
            
            if feature_vector is not None:
                features.append(feature_vector)
                processed_images.append(f"{category}/{img_name}")
            else:
                skipped_images.append(f"{category}/{img_name}")

# Convert to NumPy Array
if len(features) == 0:
    raise RuntimeError("❌ No valid features extracted! Check dataset and preprocessing.")

features_array = np.array(features)
np.save(output_feature_path, features_array)
print(f"✅ Raw Features saved at: {output_feature_path} ({len(features)} images processed)")

# Log skipped images
if skipped_images:
    with open(output_log_path, "w") as f:
        f.write("\n".join(skipped_images))
    print(f"⚠️ {len(skipped_images)} images skipped. See {output_log_path} for details.")

# Apply PCA if >1 image
if features_array.shape[0] > 1:
    n_components = min(64, features_array.shape[1])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_array)
    np.save(output_pca_feature_path, features_pca)
    print(f"✅ PCA Features saved at: {output_pca_feature_path}")
else:
    print(f"⚠️ Not enough samples for PCA. Skipping transformation.")
