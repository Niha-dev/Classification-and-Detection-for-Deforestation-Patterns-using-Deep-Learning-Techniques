import tensorflow as tf
from tensorflow.keras.models import load_model

# Define Dice Loss if used in training
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Load the trained U-Net model
unet_model_path = "models/Unet.keras"  # Update with your model's path
custom_objects = {"dice_loss": dice_loss}
unet_model = load_model(unet_model_path, custom_objects=custom_objects)
