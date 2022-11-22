"""Module for creating a deep learning model, that can be trained"""

import pathlib

import tensorflow as tf


data_dir = pathlib.Path("data/images/train/final_trainingset/")
checkpoint_path = pathlib.Path(
    "passport_mrz_reader/deep_learning/final_model/{epoch}/"
)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

normalized_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)
image_batch, labels_batch = next(iter(normalized_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

NUM_CLASSES = 37

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES),
    ]
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=False, verbose=1
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    callbacks=checkpoint_callback,
)
