import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential

from readml.explainers.dl.max_activations import MaxActivation

HEIGHT = 200
WIDTH = 200
NB_CHANNEL_CONV2D = [2, 4]
INPUT_IMAGE_3x3 = np.array([[255, 255, 0], [0, 255, 0], [0, 255, 255]])


def get_model(nb_channel):
    model = Sequential(
        [
            Input(shape=(HEIGHT, WIDTH, nb_channel)),
            Conv2D(
                NB_CHANNEL_CONV2D[0],
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                padding="same",
                name="conv2d",
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                NB_CHANNEL_CONV2D[1],
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                padding="valid",
                name="conv2d_1",
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(
                1,
                activation="sigmoid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
            ),
        ]
    )
    model.weights[0][:, :, 0, 1].assign(np.array([[1, 1, -1], [0, 1, -1], [0, 1, 1]]))
    model.weights[2][:, :, 0, 0].assign(np.zeros((3, 3)))
    model.weights[2][:, :, 1, 0].assign(
        np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
    )
    return model


def get_image_dataset(nb_channel):
    np.random.seed(0)
    image_dataset = np.random.randint(0, 255, size=(10, HEIGHT, WIDTH, nb_channel))
    image_dataset[0, 5:8, 10:13, 0] = INPUT_IMAGE_3x3
    img_dataset = tf.data.Dataset.from_tensor_slices(image_dataset)
    img_dataset = img_dataset.batch(3)
    return img_dataset


def test_get_filter_max_activations_from_image_dataset():
    nb_channel = 1
    model = get_model(nb_channel)
    img_dataset = get_image_dataset(nb_channel)
    max_activation = MaxActivation(model)
    filter_max_activations = (
        max_activation.get_filter_max_activations_from_image_dataset(img_dataset)
    )
    assert len(filter_max_activations) == 2
    assert len(filter_max_activations["conv2d"]) == NB_CHANNEL_CONV2D[0]
    assert len(filter_max_activations["conv2d_1"]) == NB_CHANNEL_CONV2D[1]
    assert filter_max_activations["conv2d"][1]["activation"][1] == np.sum(
        INPUT_IMAGE_3x3
    )
    assert (
        filter_max_activations["conv2d"][1]["input"][1].numpy()[:, :, 0]
        == INPUT_IMAGE_3x3
    ).all()
    assert filter_max_activations["conv2d_1"][0]["activation"][2] == 719
    np.testing.assert_array_equal(
        filter_max_activations["conv2d_1"][0]["input"][2].numpy()[:, :, 0],
        np.array(
            [
                [9, 188, 91, 111, 163, 83, 76, 18],
                [238, 165, 171, 211, 88, 70, 148, 134],
                [211, 45, 54, 255, 255, 0, 134, 219],
                [168, 150, 30, 0, 255, 0, 106, 74],
                [135, 29, 78, 0, 255, 255, 16, 60],
                [10, 145, 131, 231, 73, 29, 195, 199],
                [42, 202, 37, 106, 40, 111, 27, 132],
                [242, 178, 86, 181, 10, 43, 187, 158],
            ]
        ),
    )

    nb_channel = 3
    model = get_model(nb_channel)
    img_dataset = get_image_dataset(nb_channel)
    max_activation = MaxActivation(model)
    filter_max_activations = (
        max_activation.get_filter_max_activations_from_image_dataset(img_dataset)
    )
    assert filter_max_activations["conv2d"][0]["input"][0].shape == np.array([3, 3, 3])


def test_create_input_image_maximizing_activations():
    np.random.seed(0)
    nb_channel = 1
    model = get_model(nb_channel)
    max_activation = MaxActivation(model)
    (
        input_value,
        activation_history,
        grad_history,
    ) = max_activation.create_input_image_maximizing_activations(
        "conv2d",
        1,
        True,
        [0, 255],
        100,
        nb_activation_to_max="single",
        learning_rate=0.01,
        change_relu=True,
    )
    expected_output = np.array(
        [
            [[254.01408], [253.98782], [1.0375116]],
            [[104.22838], [253.97734], [1.0404677]],
            [[90.436195], [253.95912], [253.99373]],
        ]
    )
    np.testing.assert_array_almost_equal(input_value, expected_output, decimal=4)
    assert np.all(np.diff(activation_history) > 0)
