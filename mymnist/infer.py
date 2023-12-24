import fire
import numpy as np
import tensorflow as tf


# Random seed for reproducibility
seed = 42

tf.keras.utils.set_random_seed(seed)

# Save the model at the end?
save_model = True

# Batch sizes for training and testing
batch_size = 64
test_batch_size = 14

# Training epochs (usually 10 is a good value)
n_epochs = 2

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.7

# Number of target classes in the MNIST data
num_classes = 10

# Data input shape
input_shape = (28, 28, 1)


def infer(X_test_name: str):
    # The scaled mean and standard deviation of the MNIST dataset (precalculated)
    data_mean = 0.1307
    data_std = 0.3081

    # Load the MNIST dataset
    x_test = np.load(X_test_name)
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

    model.load_weights("../data/mnist_cnn_tf.ckpt").expect_partial()
    pred = model.predict(x_test)

    print(pred)


if __name__ == "__main__":
    fire.Fire(infer)
