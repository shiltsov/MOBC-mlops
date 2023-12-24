import fire
import hydra
import numpy as np
import tensorflow as tf
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


# Data input shape
input_shape = (28, 28, 1)
num_classes = 10


@hydra.main(config_path="../configs", config_name="mobc-mlops", version_base="1.3")
def infer(cfg: DictConfig = None) -> None:
    DVCFileSystem().get("../data/X_test.npy", "../data/X_test.npy")
    x_test = np.load("../data/X_test.npy")

    # The scaled mean and standard deviation of the MNIST dataset (precalculated)
    data_mean = 0.1307
    data_std = 0.3081

    # Load the MNIST dataset

    # y_test = np.load(y_test_name)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Normalize the data
    x_test = (x_test / 255.0 - data_mean) / data_std
    # y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.Conv2D(
                64, (3, 3), strides=(1, 1), padding="valid", activation="relu"
            ),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.load_weights("../models/mnist_cnn_tf.ckpt").expect_partial()
    pred = model.predict(x_test)

    print(pred)


if __name__ == "__main__":
    fire.Fire(infer)
