import logging
import pathlib

import fire
import git
import hydra
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="mobc-mlops", version_base="1.3")
def train(cfg: DictConfig = None) -> None:
    data_mean = cfg.data.data_mean
    data_std = cfg.data.data_std

    # Load the MNIST dataset
    cdir = pathlib.Path(__file__).parents[1].joinpath("data")

    DVCFileSystem().get("/data/X_train.csv", str(cdir.joinpath("X_train.csv")))
    DVCFileSystem().get("/data/y_train.csv", str(cdir.joinpath("y_train.csv")))
    DVCFileSystem().get("/data/X_test.csv", str(cdir.joinpath("X_test.csv")))
    DVCFileSystem().get("/data/y_test.csv", str(cdir.joinpath("y_test.csv")))

    x_train = np.loadtxt(cdir.joinpath("X_train.csv"), delimiter=",")
    x_train = x_train.reshape(
        x_train.shape[0], cfg.data.input_shape_x, cfg.data.input_shape_y
    )
    y_train = np.loadtxt(cdir.joinpath("y_train.csv"), delimiter=",")

    x_test = np.loadtxt(cdir.joinpath("X_test.csv"), delimiter=",")
    x_test = x_test.reshape(
        x_test.shape[0], cfg.data.input_shape_x, cfg.data.input_shape_y
    )
    y_test = np.loadtxt(cdir.joinpath("y_test.csv"), delimiter=",")

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = (x_train / 255.0 - data_mean) / data_std

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = (x_test / 255.0 - data_mean) / data_std

    y_train = tf.one_hot(y_train.astype(np.int32), depth=cfg.data.num_classes)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=cfg.data.num_classes)

    shape = (
        cfg.data.input_shape_x,
        cfg.data.input_shape_y,
        cfg.data.input_shape_z,
    )

    mlflow.set_tracking_uri(uri=cfg.mlflow.uri)

    with mlflow.start_run():
        # логируем коммит
        hexsha = git.Repo(search_parent_directories=True).head.object.hexsha
        mlflow.log_param("commit_id", hexsha)

        # логируем параметры
        data_params = {f"data_{param}": value for param, value in cfg.data.items()}
        model_parans = {f"model_{param}": value for param, value in cfg.model.items()}
        training_params = {
            f"train_{param}": value for param, value in cfg.training.items()
        }

        mlflow.log_params(data_params)
        mlflow.log_params(model_parans)
        mlflow.log_params(training_params)
        logging.info("Parameters logged")

        mlflow.keras.autolog()

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
                tf.keras.layers.Dense(cfg.data.num_classes, activation="softmax"),
            ]
        )

        # Decay the learning rate at a base rate of gamma roughly every epoch, which
        # is len(x_train) steps
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.training.learning_rate,
            decay_steps=len(x_train),
            decay_rate=cfg.training.gamma,
        )

        # Define the optimizer to user for gradient descent
        optimizer = tf.keras.optimizers.Adadelta(scheduler)

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "acc",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        # Display a model summary
        logging.info(model.summary())

        # Train the model
        model.fit(
            x_train,
            y_train,
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.n_epochs,
            validation_data=(x_test, y_test),
            validation_batch_size=cfg.training.test_batch_size,
        )

        if cfg.training.save_model:
            cdir = pathlib.Path(__file__).parents[1].joinpath("models")
            modelfile = cdir.joinpath(cfg.training.save_weights_file)
            model.save(modelfile)

        logging.info("Model ready")


if __name__ == "__main__":
    fire.Fire(train)
