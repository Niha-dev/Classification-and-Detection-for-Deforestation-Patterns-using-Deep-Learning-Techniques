from flask import Flask, render_template, request, redirect, url_for, flash, session , jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import os
import time
import logging
import numpy as np
import joblib
import cv2
import tensorflow as tf
from datetime import datetime, date
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
from dotenv import load_dotenv

app = Flask(__name__)
app.config['BASE_DIR'] = "C:/Users/nniha/flask_env/Scripts/project/"
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Access the values
BASE_DIR = app.config['BASE_DIR']
SECRET_KEY = app.config['SECRET_KEY']
print("SECRET_KEY:", repr(os.getenv('SECRET_KEY')))

# MongoDB Connection
from pymongo import MongoClient

def get_db_connection():
    try:
        client = MongoClient(
            "mongodb+srv://niharika30161007:Niharika1234@cluster0.2ojy4.mongodb.net/UserRegistrationDB?retryWrites=true&w=majority&appName=Cluster0",
            serverSelectionTimeoutMS=5000  # 5 seconds
        )
        db = client.UserRegistrationDB  # Use the correct case: UserRegistrationDB
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

try:
    client = MongoClient(
        "mongodb+srv://niharika30161007:Niharika1234@cluster0.2ojy4.mongodb.net/UserRegistrationDb?retryWrites=true&w=majority&appName=Cluster0",
        serverSelectionTimeoutMS=5000  # 5 seconds
    )
    client.server_info()  # Force a connection attempt
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")


# Test connection



# ✅ Paths
BASE_DIR = os.getenv("BASE_DIR", "C:/Users/nniha/flask_env/Scripts/project/")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads/")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static/results/")
UNET_MODEL_PATH = os.path.join(BASE_DIR, "models/Unet.keras")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models/svm_model.pkl")
PCA_MODEL_PATH = os.path.join(BASE_DIR, "models/pca_mod.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# ✅ Flask Config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# ✅ Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "visualizations"), exist_ok=True)

# ✅ Load Models
logging.info("🔄 Loading Models...")
try:
    unet_model = load_model(UNET_MODEL_PATH, compile=False)
    svm_model = joblib.load(SVM_MODEL_PATH)
    pca_model = joblib.load(PCA_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Models loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    exit(1)

# ✅ Identify Feature Extraction Layer Dynamically
def find_feature_layer(model):
    """Finds a convolutional layer with the highest number of output channels."""
    feature_layer = None
    max_channels = 0
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            channels = layer.output.shape[-1]
            if channels > max_channels:
                max_channels = channels
                feature_layer = layer
    if feature_layer:
        logging.info(f"✅ Using Feature Extraction Layer: {feature_layer.name}")
        logging.info(f"✅ Feature Layer Output Shape: {feature_layer.output.shape}")
        return feature_layer
    raise ValueError("❌ No convolutional layer found for feature extraction!")

FEATURE_LAYER = find_feature_layer(unet_model)

# ✅ Image Preprocessing
def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocesses the image for model input."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Ensure 3 channels
    if img is None:
        logging.error(f"❌ Error: Unable to read image {img_path}")
        return None
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

# ✅ Feature Extraction
def extract_features(img_array, model):
    """Extracts features from the U-Net model."""
    try:
        feature_extractor = Model(inputs=model.input, outputs=FEATURE_LAYER.output)
        features = feature_extractor(img_array, training=False)
        pooled_features = GlobalAveragePooling2D()(features).numpy().flatten()
        logging.info(f"✅ Extracted Features Shape: {pooled_features.shape}")
        return pooled_features
    except Exception as e:
        logging.error(f"❌ Error during feature extraction: {e}")
        return None

# ✅ Visualizations for Debugging
def visualize_raw_mask(mask, title="Raw Mask", filename="raw_mask.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.colorbar(label="Pixel Intensity")
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "visualizations", filename))
    plt.close()

def visualize_preprocessed_image(img_array, title="Preprocessed Image", filename="preprocessed.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array[0])  # img_array is (1, height, width, channels)
    plt.title(title)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "visualizations", filename))
    plt.close()

def visualize_segmented_image(segmented_image, title="Segmented Image", filename="segmented.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.title(title)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "visualizations", filename))
    plt.close()

def compare_features(features1, features2, title1="Features 1", title2="Features 2", filename="features.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(features1)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.plot(features2)
    plt.title(title2)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "visualizations", filename))
    plt.close()

# ✅ Deforestation Detection Pipeline

import cv2
import numpy as np
import os
import logging

def detect_deforestation(img_path, filename):
    """Perform deforestation detection and classify the type of deforestation."""
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"❌ Error: Unable to read image {img_path}")
        return "Error: Image could not be processed", None

    # ✅ Define BGR color ranges
    lower_green = np.array([0, 50, 0])
    upper_green = np.array([100, 255, 100])

    lower_blue = np.array([50, 0, 0])
    upper_blue = np.array([255, 100, 100])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([40, 40, 40])  # Adjusted black threshold

    # ✅ Create individual masks
    green_mask = cv2.inRange(img, lower_green, upper_green)
    blue_mask = cv2.inRange(img, lower_blue, upper_blue)
    black_mask = cv2.inRange(img, lower_black, upper_black)

    # ✅ Combine non-deforested masks (green, blue, black)
    non_deforested_mask = cv2.bitwise_or(green_mask, blue_mask)
    non_deforested_mask = cv2.bitwise_or(non_deforested_mask, black_mask)

    # ✅ Invert the mask to get deforested areas
    deforested_mask = cv2.bitwise_not(non_deforested_mask)

    # ✅ Debugging - Save masks for verification
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "black_mask.png"), black_mask)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "non_deforested_mask.png"), non_deforested_mask)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "deforested_mask.png"), deforested_mask)

    # ✅ Count pixels
    total_pixels = img.shape[0] * img.shape[1]
    non_deforested_pixels = cv2.countNonZero(non_deforested_mask)
    deforested_pixels = cv2.countNonZero(deforested_mask)

    # ✅ Calculate percentages
    non_deforested_percentage = (non_deforested_pixels / total_pixels) * 100
    deforested_percentage = (deforested_pixels / total_pixels) * 100

    # ✅ Determine classification
    if non_deforested_percentage > 50:
        classification_color = "Green"
        predicted_label = "Non-Deforested"
        deforestation_type = classify_deforestation_type(deforested_mask)
    else:
        classification_color = "Red"
        predicted_label = "Deforested"
        deforestation_type = classify_deforestation_type(deforested_mask)

    # ✅ Create segmented image
    segmented_image = np.zeros_like(img)
    segmented_image[non_deforested_mask != 0] = [0, 255, 0]  # Green for non-deforested
    segmented_image[deforested_mask != 0] = [0, 0, 255]  # Red for deforested

    # ✅ Save segmented image
    segmented_filename = f"seg_{filename}"
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
    cv2.imwrite(segmented_path, segmented_image)

    # ✅ Debugging visualization
    visualize_segmented_image(segmented_image, title=f"Segmented Image for {filename}", filename=f"segmented_{filename}")

    return (
        classification_color,
        predicted_label,
        segmented_filename,
        round(deforested_percentage, 2),
        round(non_deforested_percentage, 2),
        deforestation_type
    )


def classify_deforestation_type(deforested_mask):
    """Classify the type of deforestation based on the deforested mask characteristics."""
    contours, _ = cv2.findContours(deforested_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "No Deforestation Detected"

    areas = [cv2.contourArea(contour) for contour in contours]
    perimeters = [cv2.arcLength(contour, closed=True) for contour in contours]
    
    total_deforested_area = np.sum(areas)
    avg_area = np.mean(areas)
    max_area = np.max(areas)

    # ✅ **Clear-Cutting:** Large continuous deforested area
    if max_area > 0.6 * total_deforested_area:
        return "Clear-Cutting"

    # ✅ **Selective Logging:** Many small scattered patches
    if avg_area < 1000 and len(contours) > 20:
        return "Selective Logging"

    # ✅ **Forest Degradation:** Irregular deforestation edges
    irregularity = np.mean([p / (a + 1e-5) for p, a in zip(perimeters, areas)])
    if irregularity > 0.1:
        return "Forest Degradation"

    # ✅ **Burned Area:** Low-intensity, widespread deforestation
    if np.median(areas) < 2000 and len(contours) > 5:
        return "Burned Area"

    # ✅ **Road Expansion:** Long narrow deforested paths
    aspect_ratios = [cv2.boundingRect(contour)[2] / (cv2.boundingRect(contour)[3] + 1e-5) for contour in contours]
    if np.mean(aspect_ratios) > 5:
        return "Road Expansion"

    # ✅ **Agricultural Expansion:** Large uniform patches
    if np.std(areas) < 5000 and total_deforested_area > 50000:
        return "Agricultural Expansion"

    return "Unknown"

# ✅ Serve Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/planting')
def planting():
    return render_template('planting.html')

@app.route('/recylce')
def recylce():
    return render_template('recylce.html')

@app.route('/water_saving')
def water_saving():
    return render_template('water_saving.html')

@app.route('/tree_saving')
def tree_saving():
    return render_template('tree_saving.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/animal_saving')
def animal_saving():
    return render_template('animal_saving.html')

@app.route('/solar_panel')
def solar_panel():
    return render_template('solar_panel.html')
# Index (Dashboard) Route after login

@app.route("/index")
def index():
    if "user" not in session:
        flash("⚠️ Please log in first!", "warning")
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])

# Registration Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get("firstName")
        last_name = request.form.get("lastName")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirmPassword")
        dob = request.form.get("dob")
        gender = request.form.get("gender")
        country = request.form.get("country")
        terms = request.form.get("terms")

        if not all([first_name, last_name, email, password, confirm_password, dob, gender, country, terms]):
            flash("❌ Please fill out all fields!", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("❌ Passwords do not match!", "danger")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        # Connect to MongoDB
        db = get_db_connection()
        users_collection = db["users"]

        # Check if email already exists
        if users_collection.find_one({"email": email}):
            flash("❌ Email already registered!", "danger")
            return redirect(url_for("register"))

        # Insert new user into MongoDB
        users_collection.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": hashed_password,
            "dob": dob,
            "gender": gender,
            "country": country
        })

        flash("✅ Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        db = get_db_connection()
        if db is None:
            flash("❌ Database connection failed. Please try again.", "danger")
            return redirect(url_for("login"))

        users_collection = db["users"]
        user = users_collection.find_one({"email": email})

        if user and check_password_hash(user["password"], password):
            session["user"] = email
            flash("✅ Login successful! Welcome back.", "success")
            return redirect(url_for("index"))
        else:
            flash("❌ Invalid email or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        dob = request.form.get('dob')
        print(f"Received DOB: {dob}")  # Debugging
        return redirect(url_for('reset_password'))  # Change this as needed
    return render_template('forgot.html')




@app.route('/reset', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        new_password = request.form.get('new_password')

        if not new_password:
            flash("❌ Password cannot be empty!", "danger")
            return redirect(url_for('reset_password'))

        db = get_db_connection()
        users_collection = db["users"]

        # Update the user's password (You may need to fetch the user session info)
        users_collection.update_one(
            {"dob": session.get("dob")},  # Use the DOB stored in session
            {"$set": {"password": generate_password_hash(new_password)}}
        )

        flash("✅ Password reset successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('reset.html')


# Logout Route
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("✅ Logged out successfully!", "success")
    return redirect(url_for("home"))


# ✅ Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash("No file uploaded!", "error")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file!", "error")
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Perform deforestation detection
    result = detect_deforestation(filepath, filename)
    if result[1] is None:
        flash("Prediction failed! Check logs for details.", "error")
        return redirect(request.url)

    # 🧠 Add suggestion logic based on deforestation type
    deforestation_type = result[5]
    suggestions_dict = {
    "Clear-Cutting": "Large-scale forest loss detected. Enforce logging limits and promote reforestation.",
    
    "Selective Logging": "Small patch removals observed. Monitor forest health and restore canopy cover.",
    
    "Forest Degradation": "Signs of ecosystem stress. Improve land use and protect biodiversity zones.",
    
    "Burned Area": "Burned areas found. Apply fire control and restore native vegetation.",
    
    "Road Expansion": "Infrastructure expansion detected. Use green planning and protect forest borders.",
    
    "Agricultural Expansion": "Farming activity spotted. Promote agroforestry and regulate land conversion.",
    
    "Unknown": "Pattern unclear. Suggest manual review and better imagery or model refinement.",
    
    "No Deforestation Detected": "Forest appears stable. Continue regular monitoring."
}


    suggestion = suggestions_dict.get(deforestation_type, "No specific suggestion available.")

    # Pass everything to the template
    return render_template(
        "result.html",
        uploaded_image=filename,
        segmented_image=result[2],
        predicted_class=result[1],
        classification_color=result[0],
        deforested_percentage=result[3],
        non_deforested_percentage=result[4],
        deforestation_type=deforestation_type,
        suggestion=suggestion  # ✅ This is the 3rd point you asked about
    )


# Run the Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port,debug=True)
