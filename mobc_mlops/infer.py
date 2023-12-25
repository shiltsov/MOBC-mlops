import logging
import os

import fire
import hydra
import numpy as np
import tensorflow as tf
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="mobc-mlops", version_base="1.3")
def infer(cfg: DictConfig = None) -> None:
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")
    dvcfile = os.path.join(cdir, "X_test.npy")

    DVCFileSystem().get("/data/X_test.npy", dvcfile)
    x_test = np.load(dvcfile)

    # The scaled mean and standard deviation of the MNIST dataset (precalculated)
    data_mean = 0.1307
    data_std = 0.3081

    input_shape = (
        cfg.data.input_shape_x,
        cfg.data.input_shape_y,
        cfg.data.input_shape_z,
    )

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Normalize the data
    x_test = (x_test / 255.0 - data_mean) / data_std

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
            tf.keras.layers.Dense(cfg.data.num_classes, activation="softmax"),
        ]
    )

    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/")
    modelfile = os.path.join(cdir, cfg.training.save_weights_file)

    model.load_weights(modelfile).expect_partial()
    pred = model.predict(x_test)

    np.savetxt(cfg.infer.output_file, pred)
    logging.info("Done")


if __name__ == "__main__":
    fire.Fire(infer)
