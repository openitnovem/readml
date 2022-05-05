import itertools
import os

import cv2
import numpy as np

from readml.explainers.dl.explain_dl import ExplainDL
from readml.logger import ROOT_DIR
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

model_path = os.path.join(ROOT_DIR, "../outputs/tests/dl/model/")

def initialize_directories_dl(out_path, dir_to_create):
    os.chdir(ROOT_DIR)
    new_root = os.getcwd()
    new_root = "/".join(new_root.split("/")[:-1])
    os.chdir(new_root)
    start = out_path.index("/") + 1
    split = out_path[start:].split("/")
    for elt in split:
        if not os.path.isdir(elt):
            os.makedirs(elt)
            os.chdir(elt)
        else:
            os.chdir(elt)
    os.chdir(ROOT_DIR)

    for elt in dir_to_create:
        if not os.path.isdir(os.path.join(out_path, elt)):
            os.makedirs(os.path.join(out_path, elt))


def create_dir_image():
    dir_to_create = ["data_image", "image"]
    out_path = "../outputs/tests/dl"
    initialize_directories_dl(out_path, dir_to_create)


create_dir_image()
data_image_path = os.path.join(ROOT_DIR, "../outputs/tests/dl", "data_image")
output_path_image_dir = os.path.join(ROOT_DIR, "../outputs/tests/dl", "image")


def save_image_data(X_train, name):  # data_image_path
    for idx, img in enumerate(X_train):
        cv2.imwrite(os.path.join(data_image_path, f"{name}_{idx+1}.jpg"), img)


def test_explain_image_rgb():
    if os.listdir(output_path_image_dir) != []:
        for files in os.listdir(output_path_image_dir):
            os.remove(os.path.join(output_path_image_dir, files))

    dl_explain_image_rgb()  # data_image_path, output_path_image_dir
    min_obs, max_obs = 1, 10
    output_path_local_min_obs = os.path.join(
        output_path_image_dir, f"grad_cam_cifar_{min_obs}.jpg"
    )
    output_path_local_max_obs = os.path.join(
        output_path_image_dir, f"grad_cam_cifar_{max_obs}.jpg"
    )
    outside_output = os.path.join(
        output_path_image_dir, f"grad_cam_cifar_{max_obs + 1}.jpg"
    )

    assert os.path.isfile(output_path_local_min_obs)
    assert os.path.isfile(output_path_local_max_obs)
    assert not os.path.isfile(outside_output)


def dl_explain_image_rgb():  # data_image_path, output_path_image_dir
    X_train, y_train, width, height, channel = create_image_data_rgb()
    save_image_data(X_train, "cifar")

    model = simple_model_image_rgb(X_train, y_train, width, height, channel)

    # To test interpret in readml/main.py with image case
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)

    exp = ExplainDL(
        model=model,
        out_path=output_path_image_dir,
    )
    exp.explain_image(
        image_dir=data_image_path,
        size=(width, height),
        color_mode="rgb",
    )


def create_image_data_rgb():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    _, width, height, channel = train_images.shape
    # Focus on two labels
    focus = list(itertools.chain(*train_labels))
    focus_0_1 = [
        index for index, value in enumerate(focus) if value == 0 or value == 1
    ][0:10]
    train_images = train_images[focus_0_1]
    train_labels = np.array(
        [elt[0] for idx, elt in enumerate(train_labels) if idx in focus_0_1]
    )
    return train_images, train_labels, width, height, channel


def simple_model_image_rgb(X_train, y_train, width, height, channel):
    baseModel = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(width, height, channel)),
    )
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(1, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train)
    return model


def test_explain_image_grayscale():
    if os.listdir(output_path_image_dir) != []:
        for files in os.listdir(output_path_image_dir):
            os.remove(os.path.join(output_path_image_dir, files))

    dl_explain_image_grayscale()  # data_image_path, output_path_image_dir
    min_obs, max_obs = 1, 10
    output_path_local_min_obs = os.path.join(
        output_path_image_dir, f"grad_cam_mnist_{min_obs}.jpg"
    )
    output_path_local_max_obs = os.path.join(
        output_path_image_dir, f"grad_cam_mnist_{max_obs}.jpg"
    )
    outside_output = os.path.join(
        output_path_image_dir, f"grad_cam_mnist_{max_obs + 1}.jpg"
    )

    assert os.path.isfile(output_path_local_min_obs)
    assert os.path.isfile(output_path_local_max_obs)
    assert not os.path.isfile(outside_output)


def dl_explain_image_grayscale():  # data_image_path, output_path_image_dir
    X_train, y_train, width, height = create_image_data_grayscale()
    save_image_data(X_train, "mnist")
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    model = simple_model_image_grayscale(X_train, y_train, width, height)
    exp = ExplainDL(
        model=model,
        out_path=output_path_image_dir,
    )
    exp.explain_image(
        image_dir=data_image_path,
        size=(width, height),
        color_mode="grayscale",
    )


def create_image_data_grayscale():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    _, width, height = train_images.shape
    return train_images[0:10], train_labels[0:10], width, height


def simple_model_image_grayscale(
    X_train,
    y_train,
    width,
    height,
    channel=1,
):
    model = Sequential(
        [
            Input(shape=(width, height, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(X_train, y_train)
    return model
