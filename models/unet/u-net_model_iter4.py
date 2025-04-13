import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore

import wandb
from wandb.integration.keras import WandbMetricsLogger

from util import preprocess_images

batch_size = 16
epochs = 70

wandb.init(
    entity="mehher_ghevandiani-american-university-of-armenia",
    project="Capstone",
    config={
        "learning_rate": 0.001,
        "architecture": "U-Net",
        "dataset": "human_detection_dataset",
        "epochs": epochs,
        "batch_size": batch_size,
    },
)

print("WandB initiated")


def unet_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)  # Normalize grayscale input

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    # Final output: 2 channels (AB), use 'tanh' if your AB values are scaled to [-1, 1]
    outputs = layers.Conv2D(2, (1, 1), activation='tanh')(c9)

    model = Model(inputs, outputs)
    return model


def train_model(data_dir, callbacks=None):
    model = unet_model()
    model.compile(optimizer='adam', loss='mae', metrics=[MeanSquaredError(), MeanAbsoluteError()]) # check MAE, check imports

    images, labels = preprocess_images(data_dir)

    model.fit(
        images, labels,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    model.save(f'u_net_{epochs}_epoch_iter_4.h5')

dataset_dir = "data/cctv_data"

checkpoint_callback = ModelCheckpoint(
    filepath='u_net_colorization_checkpoint.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True 
)

callbacks = [checkpoint_callback, WandbMetricsLogger()]

with tf.device('/GPU:0'):
    train_model(dataset_dir, callbacks=callbacks)
