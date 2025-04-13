import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_and_plot_lab_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = cv2.resize(img, img_size)

        l_channel = img[:, :, 0:1] / 255.0
        ab_channels = img[:, :, 1:] / 128.0

        # Plotting
        plt.figure(figsize=(12, 4))

        # L channel
        plt.subplot(1, 3, 1)
        plt.imshow(l_channel[:, :, 0], cmap='gray')
        plt.title("Normalized L Channel")
        plt.axis("off")

        # A channel
        plt.subplot(1, 3, 2)
        plt.imshow(ab_channels[:, :, 0], cmap='gray')
        plt.title("Normalized A Channel")
        plt.axis("off")

        # B channel
        plt.subplot(1, 3, 3)
        plt.imshow(ab_channels[:, :, 1], cmap='gray')
        plt.title("Normalized B Channel")
        plt.axis("off")

        plt.tight_layout()
        # Plot all normalized LAB channels together
        plt.figure(figsize=(8, 8))
        combined_image = np.concatenate((l_channel, ab_channels), axis=2)
        plt.imshow(combined_image)
        plt.title("Normalized LAB Channels Combined")
        plt.axis("off")
        plt.show()

        return l_channel, ab_channels
    else:
        raise ValueError(f"Unable to read image at path: {image_path}")


preprocess_and_plot_lab_image("/Users/mehherghevandiani/Documents/personal/capstone/data/paintings/images/images/Diego_Velazquez/Diego_Velazquez_2.jpg")