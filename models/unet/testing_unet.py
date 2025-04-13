import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import matplotlib.pyplot as plt

from util import lab_to_rgb, preprocess_image

# Load the trained U-Net model
model = load_model("u_net_70_epoch_iter_4.h5", compile=False)

# Function to load and preprocess an image for testing
# def preprocess_image(image_path, img_size=(256, 256)):
#     img = load_img(image_path, target_size=img_size)
#     img = img_to_array(img)
#     original_image = img / 255.0  # Normalize the original image to [0, 1]
#     lab_image = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
#     l_channel = lab_image[:, :, 0:1] # / 100.0  # Normalize L channel to [0, 1] (LAB L channel is in [0, 100])
#     return original_image, l_channel



# Function to postprocess and convert LAB image to RGB
# def lab_to_rgb(l_channel, ab_channels):
#     l_channel = (l_channel * 100).astype(np.uint8)  # Scale L channel back to [0, 100]
#     ab_channels = (ab_channels * 128).astype(np.int8)  # Scale AB channels back to [-128, 127]
#     lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
#     rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
#     return rgb_image

# Python
# def lab_to_rgb(l_channel, ab_channels):
#     l_channel = (l_channel * 100).astype(np.uint8)  # Scale L channel back to [0, 100]
#     ab_channels = (ab_channels * 128).astype(np.uint8)  # Scale AB channels back to [0, 255]
#     lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
#     rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
#     return rgb_image
# Path to the test image
image_path = "/Users/mehherghevandiani/Documents/personal/capstone/data/cctv_data/human0.png"  # Replace with the path to your test image

original_image = load_img(image_path)
original_image = img_to_array(original_image) / 255.0  # Normalize the original image to [0, 1]
# Preprocess the image
l_channel, ab_channels = preprocess_image(image_path)

# Predict the AB channels using the model
l_channel_input = np.expand_dims(l_channel, axis=0)  # Add batch dimension
predicted_ab_channels = model.predict(l_channel_input)[0]  # Remove batch dimension

# Postprocess the predicted LAB image to RGB
colorized_image = lab_to_rgb(l_channel, predicted_ab_channels)

# Convert the grayscale image for display
grayscale_image = l_channel[:, :, 0]  # Scale back to [0, 100] for visualization

# Display the original, grayscale, and colorized images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale Image")
plt.imshow(grayscale_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Colorized Image")
plt.imshow(colorized_image)
plt.axis("off")

plt.show()