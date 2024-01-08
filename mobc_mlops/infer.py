import logging
import pathlib

import fire
import hydra
import numpy as np
import tensorflow as tf
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="mobc-mlops", version_base="1.3")
def infer(cfg: DictConfig = None) -> None:
    cdir = pathlib.Path(__file__).parents[1].joinpath("data")
    DVCFileSystem().get("/data/X_test.csv", str(cdir.joinpath("X_test.csv")))

    x_test = np.loadtxt(cdir.joinpath("X_test.csv"), delimiter=",")
    x_test = x_test.reshape(
        x_test.shape[0], cfg.data.input_shape_x, cfg.data.input_shape_y
    )

    data_mean = cfg.data.data_mean
    data_std = cfg.data.data_std

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Normalize the data
    x_test = (x_test / 255.0 - data_mean) / data_std

    cdir = pathlib.Path(__file__).parents[1].joinpath("models")
    modelfile = cdir.joinpath(cfg.training.save_weights_file)

    model = tf.keras.models.load_model(modelfile)
    pred = model.predict(x_test)

    np.savetxt(cfg.infer.output_file, pred)
    logging.info("Done")


if __name__ == "__main__":
    fire.Fire(infer)
