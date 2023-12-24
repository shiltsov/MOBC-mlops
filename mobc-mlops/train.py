import fire
import hydra
import numpy as np
import tensorflow as tf
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="mobc-mlops", version_base="1.3")
def train(cfg: DictConfig = None) -> None:
    data_mean = 0.1307
    data_std = 0.3081

    # Load the MNIST dataset
    DVCFileSystem().get("../data/X_train.npy", "../data/X_train.npy")
    DVCFileSystem().get("../data/y_train.npy", "../data/y_train.npy")
    DVCFileSystem().get("../data/X_test.npy", "../data/X_test.npy")
    DVCFileSystem().get("../data/y_test.npy", "../data/y_test.npy")

    x_train = np.load("../data/X_train.npy")
    y_train = np.load("../data/y_train.npy")
    x_test = np.load("../data/X_test.npy")
    y_test = np.load("../data/y_test.npy")

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = (x_train / 255.0 - data_mean) / data_std

    y_train = tf.one_hot(y_train.astype(np.int32), depth=cfg["data"]["num_classes"])

    shape = (
        cfg["data"]["input_shape_x"],
        cfg["data"]["input_shape_y"],
        cfg["data"]["input_shape_z"],
    )

    # Define the architecture of the neural network
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                input_shape=shape,
            ),
            tf.keras.layers.Conv2D(
                64, (3, 3), strides=(1, 1), padding="valid", activation="relu"
            ),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(cfg["data"]["num_classes"], activation="softmax"),
        ]
    )

    # Decay the learning rate at a base rate of gamma roughly every epoch, which
    # is len(x_train) steps
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg["training"]["learning_rate"],
        decay_steps=len(x_train),
        decay_rate=cfg["training"]["gamma"],
    )

    # Define the optimizer to user for gradient descent
    optimizer = tf.keras.optimizers.Adadelta(scheduler)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    # Display a model summary
    print(model.summary())

    # Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["n_epochs"],
        validation_data=(x_test, y_test),
        validation_batch_size=cfg["training"]["test_batch_size"],
    )

    if cfg["training"]["save_model"]:
        model.save_weights("../models/mnist_cnn_tf.ckpt")

    print("Model ready")


if __name__ == "__main__":
    fire.Fire(train)
