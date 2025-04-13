import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError # type: ignore

# Load the trained model with the specified loss function
model = load_model("./models/conv2d/colorization_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

# Function to load and preprocess a grayscale image
def load_and_preprocess_image(image_path, img_size=(128, 128)):
    img = load_img(image_path, target_size=img_size, color_mode="grayscale")
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to load the original image
def load_original_image(image_path, img_size=(128, 128)):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

# Function to postprocess and convert LAB image to RGB
def lab_to_rgb(l_channel, ab_channels):
    l_channel = (l_channel * 255).astype(np.uint8)
    ab_channels = (ab_channels * 255 - 128).astype(np.uint8)  # Scale AB channels back to [-128, 127]
    lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image

# Load and preprocess the grayscale image
image_path = "./archive/images/images/Claude_Monet/Claude_Monet_13.jpg"
image_path = "/Users/mehherghevandiani/Documents/personal/capstone/data/human_detection_dataset/human0.png"
gray_image = load_and_preprocess_image(image_path)

# Load the original image
original_image = load_original_image(image_path)

# Predict the colorized image
predicted_ab_channels = model.predict(gray_image)
predicted_ab_channels = predicted_ab_channels[0]  # Remove batch dimension

# Postprocess and convert to RGB
gray_image = gray_image[0, :, :, 0]  # Remove batch dimension and channel dimension
gray_image = np.expand_dims(gray_image, axis=-1)  # Add channel dimension for LAB conversion
colorized_image = lab_to_rgb(gray_image, predicted_ab_channels)

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