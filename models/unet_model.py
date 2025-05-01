import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import os

# ⚠️ Disable GPU explicitly (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("⚠️ GPU detected but disabled. Running on CPU.")
else:
    print("✅ No GPU detected. Running on CPU.")

def unet_resnet50(input_shape):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Load ResNet50 as the encoder (backbone)
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    # ✅ Extract feature layers from ResNet50 (Ensure correct layer names)
    try:
        skips = [
            base_model.get_layer("conv1_relu").output,
            base_model.get_layer("conv2_block3_out").output,
            base_model.get_layer("conv3_block4_out").output,
            base_model.get_layer("conv4_block6_out").output
        ]
        encoder_output = base_model.get_layer("conv5_block1_out").output
    except KeyError as e:
        print(f"⚠️ Layer not found: {e}. Using base_model.output instead.")
        encoder_output = base_model.output  # Use last ResNet layer if named layer is missing

    # ✅ Unfreeze only BatchNorm layers (helps training)
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    # Decoder block definition
    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        return x

    # Decoder blocks
    d1 = decoder_block(encoder_output, skips[3], 512)
    d2 = decoder_block(d1, skips[2], 256)
    d3 = decoder_block(d2, skips[1], 128)
    d4 = decoder_block(d3, skips[0], 64)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(d4)

    # Create the model
    model = models.Model(inputs, outputs)
    return model

# ✅ Build & Compile Model
input_shape = (128, 128, 3)
model = unet_resnet50(input_shape)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ✅ Save Model in Both Formats
save_path_keras = "C:/Users/nniha/flask_env/Scripts/project/models/Unet.keras"
save_path_h5 = "C:/Users/nniha/flask_env/Scripts/project/models/Unet.h5"

os.makedirs(os.path.dirname(save_path_keras), exist_ok=True)  # Ensure directory exists

model.save(save_path_keras)  # Save as .keras
model.save(save_path_h5)  # Save as .h5

print(f"✅ Model saved successfully at:\n - {save_path_keras}\n - {save_path_h5}")
