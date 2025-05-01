import os
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paths - Verify these are correct
dataset_path = "C:/Users/nniha/flask_env/Scripts/project/static/Images/"
npy_output_path = "C:/Users/nniha/flask_env/Scripts/project/models/labels.npy"
json_output_path = "C:/Users/nniha/flask_env/Scripts/project/models/labels.json"
feature_file = "C:/Users/nniha/flask_env/Scripts/project/models/features.npy"

# Updated categories - removed 'non-deforested' since folder is missing
categories = {
    "deforested":4,
    "non_deforested": 5,  # Old (with hyphen)
    "clear_cutting": 3,
    "edge_deforested": 2,
    "fragmented": 1,
    "selective_logging": 0
}

def validate_dataset():
    """Check dataset structure and count files"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset folder not found at {dataset_path}")
    
    total_files = 0
    valid_categories = []
    
    for category in categories:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            logger.warning(f"⚠️ Missing folder: {category}")
            continue
        
        count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if count == 0:
            logger.warning(f"⚠️ Empty folder: {category}")
            continue
            
        valid_categories.append(category)
        total_files += count
        logger.info(f"✅ {category}: {count} images")
    
    return valid_categories, total_files

def generate_labels(valid_categories):
    """Generate labels for existing categories"""
    labels = []
    label_dict = {}
    
    for category in valid_categories:
        folder = os.path.join(dataset_path, category)
        images = sorted([
            f for f in os.listdir(folder) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        label = categories[category]
        labels.extend([label] * len(images))
        label_dict.update({f"{category}/{img}": label for img in images})
    
    return np.array(labels, dtype=np.int32), label_dict

def check_feature_match(labels):
    """Verify label-feature count match"""
    if not os.path.exists(feature_file):
        logger.warning("⚠️ features.npy not found - skipping validation")
        return True
        
    features = np.load(feature_file)
    if len(labels) != features.shape[0]:
        logger.error(f"❌ Mismatch: {len(labels)} labels vs {features.shape[0]} features")
        return False
    return True

def main():
    logger.info("\n🔍 Validating dataset structure...")
    valid_categories, total_files = validate_dataset()
    
    if not valid_categories:
        raise ValueError("❌ No valid categories found")
    
    logger.info(f"\n📊 Total images found: {total_files}")
    
    logger.info("\n🏷 Generating labels...")
    labels, label_dict = generate_labels(valid_categories)
    
    if not check_feature_match(labels):
        raise ValueError("Label-feature count mismatch")
    
    # Backup existing files
    for path in [npy_output_path, json_output_path]:
        if os.path.exists(path):
            os.rename(path, path + ".bak")
    
    # Save new files
    np.save(npy_output_path, labels)
    with open(json_output_path, 'w') as f:
        json.dump(label_dict, f, indent=4)
    
    logger.info(f"\n🎉 Success! Saved {len(labels)} labels to:")
    logger.info(f"→ {npy_output_path}")
    logger.info(f"→ {json_output_path}")

if __name__ == "__main__":
    main()