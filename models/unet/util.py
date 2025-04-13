import numpy as np
import os
import cv2

def preprocess_image(img_path, img_size=(256, 256)):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        img = cv2.resize(img, img_size)
        l_channel = img[:, :, 0:1] / 255.0  # Normalize L channel
        ab_channels = (img[:, :, 1:] - 128) / 128.0  # Normalize AB channels
        return l_channel, ab_channels
    else:
        print(f"Image at {img_path} could not be read.")

def preprocess_images(data_dir, img_size=(256, 256)):
    l_channels = []
    ab_channels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            img_path = os.path.join(root, file)
            l_channel, ab_channel = preprocess_image(img_path, img_size)
            l_channels.append(l_channel)
            ab_channels.append(ab_channel)

    l_channels = np.array(l_channels, dtype=np.float32)
    ab_channels = np.array(ab_channels, dtype=np.float32)

    return l_channels, ab_channels


def lab_to_rgb(l_channel, ab_channels):
    l_channel = (l_channel * 255).astype(np.uint8)
    ab_channels = (ab_channels * 128 + 128).astype(np.uint8)
    lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image