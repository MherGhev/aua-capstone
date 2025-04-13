import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.init(
    entity="mehher_ghevandiani-american-university-of-armenia",
    project="Capstone",
    config={
        "learning_rate": 0.001,
        "architecture": "conv2d",
        "dataset": "paintings",
        "epochs": 50,
        "batch_size": 8,
    },
)

print("WandB initiated")


def load_images(path, img_size=(128, 128)):
    images = []
    for artist in os.listdir(path):
        for filename in os.listdir(path + artist):
            img = load_img(os.path.join(path, artist, filename), target_size=img_size)
            img = img_to_array(img) / 255.0
            images.append(img)
    return np.array(images)

dataset_path = "/Users/mehherghevandiani/Documents/personal/capstone/data/paintings/images/images/"
images = load_images(dataset_path)

print(images[0])

print("Images loaded")

# Convert images to LAB color space
lab_images = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in images]
l_channel = np.array([img[:, :, 0] for img in lab_images]) / 255.0  # Normalize L channel (grayscale)
ab_channels = np.array([img[:, :, 1:] for img in lab_images]) / 255.0  # Normalize A and B channels

print("Images Converted to LAB")

# Prepare dataset
X_train, X_test, Y_train, Y_test = train_test_split(l_channel, ab_channels, test_size=0.2)

X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)

# Reshape Y_train and Y_test to match the model's output shape
Y_train = np.reshape(Y_train, (Y_train.shape[0], 128, 128, 2))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 128, 128, 2))

# Build the CNN model
def build_colorization_model(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(2, (3, 3), activation="tanh", padding="same")(x)  # 2 output channels (A, B)

    model = Model(inputs, x)
    return model

# Compile and train model
model = build_colorization_model()
model.compile(optimizer="adam", loss="mse")

callbacks = [WandbMetricsLogger()]


with tf.device('/GPU:0'):
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, callbacks=callbacks)

# Save model
model.save("colorization_model.h5")

