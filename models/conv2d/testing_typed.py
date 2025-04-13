import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from typing import Tuple

# Load the trained model with the specified loss function
model: "tensorflow.keras.Model" = load_model("./models/conv2d/colorization_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

# Function to load and preprocess a grayscale image
def load_and_preprocess_image(image_path: str, img_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = load_img(image_path, target_size=img_size, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load the original image
def load_original_image(image_path: str, img_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Function to postprocess and convert LAB image to RGB
def lab_to_rgb(l_channel: np.ndarray, ab_channels: np.ndarray) -> np.ndarray:
    l_channel = (l_channel * 255).astype(np.uint8)  # Scale L channel back to [0, 255]
    ab_channels = (ab_channels * 255 - 128).astype(np.uint8)  # Scale AB channels back to [-128, 127]
    lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image

# Load and preprocess the grayscale image
image_path: str = "/Users/mehherghevandiani/Documents/personal/capstone/data/human_detection_dataset/human0.png"
gray_image: np.ndarray = load_and_preprocess_image(image_path)

# Load the original image
original_image: np.ndarray = load_original_image(image_path)

# Predict the colorized image
predicted_ab_channels: np.ndarray = model.predict(gray_image)[0]  # Remove batch dimension

# Postprocess and convert to RGB
gray_image = gray_image[0, :, :, 0]  # Remove batch dimension and channel dimension
gray_image = np.expand_dims(gray_image, axis=-1)  # Add channel dimension for LAB conversion
colorized_image: np.ndarray = lab_to_rgb(gray_image, predicted_ab_channels)

# Display the original, grayscale, and colorized images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale Image")
plt.imshow(gray_image[:, :, 0], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Colorized Image")
plt.imshow(colorized_image)
plt.axis("off")

plt.show()