import io
from collections.abc import Iterable
from itertools import tee
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class MaxActivation:
    def __init__(self, model: Model):
        """Initialize the instance with the provided model.
        Check the provided objects.

        Parameters
        ----------
        model : Model
            CNN model.
        """
        self.model = model
        self._check_model()
        self.conv2d_layers = self._get_conv2d_layers()
        self.activations_model = self._create_activations_model_from_conv2d_layers(
            self.conv2d_layers
        )

    def _check_model(self):
        """Check the provided model.

        Raises
        ------
        TypeError
            Raise an error if the provided model is not a tensorflow Model instance.

        TypeError
            Raise an error if the provided model input has more than a single input
            corresponding to a single image.
        """
        if not isinstance(self.model, Model):
            raise TypeError(
                "The provided model must be a tensorflow Model instance. "
                "This is the only model which is currently supported."
            )

        if isinstance(self.model.inputs, list) and len(self.model.inputs) > 1:
            raise ValueError(
                "The provided model must have a single input "
                "corresponding to the input image. "
            )

    def _get_conv2d_layers(self) -> List[Tuple[str, tf.Tensor]]:
        """Get all Conv2D layers of the model.

        Returns
        -------
        List[Tuple[str, tf.Tensor]]
            List of Conv2D layers with their name and the corresponding tensor.
        """
        conv2d_layers = [
            (layer.name, layer.output)
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        return conv2d_layers  # type: ignore

    def _create_activations_model_from_conv2d_layers(
        self, conv2d_layers: List[Tuple[str, tf.Tensor]]
    ) -> Model:
        """
        Create a submodel, which outputs the activations of conv2D layers.
        """
        activations_model = Model(
            self.model.inputs,
            outputs=[conv2d_layer[1] for conv2d_layer in conv2d_layers],
        )
        return activations_model

    def _initialize_output(self, activations_model: Model) -> dict:
        """Initialize the output of the get_filter_max_activations method.

        Parameters
        ----------
        activations_model : Model
            Model that outputs the activations of all Conv2D layers.

        Returns
        -------
        dict
            Initialized output.
        """
        max_activations = {}
        for layer_id in range(len(activations_model.output)):
            max_activations[layer_id] = {}
            for channel_id in range(activations_model.output[layer_id].shape[-1]):
                max_activations[layer_id][channel_id] = {"activation": [], "input": []}
        return max_activations

    def _update_max_activations(
        self,
        activations_model: Model,
        predictions: np.ndarray,
        layer_name: str,
        image_batch: np.ndarray,
        overall_max_activations: dict,
        batch_max_activations: dict,
        layer_id: int,
        channel_id: int,
        nb_images_per_filter: int,
    ) -> dict:
        """Update the max_activations dictionnary so that the nb_images_per_filter
        images with the highest activations are kept considering the current
        overall_max_activations and the candidates batch_max_activations.

        For every image of the batch, the following operations are performed:
        - If the current batch_max_activations is full (its length equals to
        nb_images_per_filter), if the candidate has a higher activation, then remove the
        lowest activation of overall_max_activations.
        - If the current batch_max_activations is not full (less than
        nb_images_per_filter), add the activation of the considered batch image.
        The activation is added. The corresponding part of the input image is computed
        and added to overall_max_activations.
        """
        for batch_max_activation in batch_max_activations:
            # if there are already nb_images_per_filter images in overall_max_activations
            # but batch_max_activation is higher than the min of overall_max_activations
            # then remove the min activation of overall_max_activations
            if len(
                overall_max_activations[layer_id][channel_id]["activation"]
            ) == nb_images_per_filter and batch_max_activation.numpy() > min(
                overall_max_activations[layer_id][channel_id]["activation"]
            ):
                index_to_remove = overall_max_activations[layer_id][channel_id][
                    "activation"
                ].index(
                    min(overall_max_activations[layer_id][channel_id]["activation"])
                )
                del overall_max_activations[layer_id][channel_id]["activation"][
                    index_to_remove
                ]
                del overall_max_activations[layer_id][channel_id]["input"][
                    index_to_remove
                ]

            # if there are less than nb_images_per_filter in overall_max_activations
            # then add batch_max_activation
            if (
                len(overall_max_activations[layer_id][channel_id]["activation"])
                < nb_images_per_filter
            ):
                overall_max_activations[layer_id][channel_id]["activation"].append(
                    batch_max_activation.numpy()
                )
                loc = tf.where(
                    predictions[layer_id][:, :, :, channel_id] == batch_max_activation
                )[
                    0
                ]  # take the first one if several
                x_min, x_max, y_min, y_max = self._feature_map_loc_to_input_loc(
                    activations_model,
                    layer_name,
                    loc[1].numpy(),
                    loc[2].numpy(),
                )
                assert x_max - x_min == y_max - y_min
                overall_max_activations[layer_id][channel_id]["input"].append(
                    image_batch[
                        loc[0].numpy(),
                        x_min : x_max + 1,
                        y_min : y_max + 1,
                        :,
                    ]
                )
        return overall_max_activations

    def get_filter_max_activations_from_image_dataset(
        self, img_dataset: Iterable, nb_images_per_filter: int = 9
    ) -> dict:
        """Given a dataset of image batches and a number of images parts to keep per
        filter, retrieve the image parts that maximize the activations of all Conv2D
        layers.

        Parameters
        ----------
        img_dataset : Iterable
            Dataset yielding batchs of images of shape (batch size, height, width, number of channels).
        nb_images_per_filter: int, optional
            Number of image parts (with the highest activations) to keep per filters,
            by default 9.

        Returns
        -------
        dict
            _description_
        """
        self._check_dataset(img_dataset)
        max_activations = self._initialize_output(self.activations_model)

        # get max activations
        for image_batch in img_dataset:
            predictions = self.activations_model.predict(image_batch)
            for layer_id in range(len(self.activations_model.output)):
                layer_name = self.conv2d_layers[layer_id][0]
                new_shape = np.prod(predictions[layer_id].shape[:3])
                for channel_id in range(
                    self.activations_model.output[layer_id].shape[-1]
                ):
                    batch_max_activations = tf.sort(
                        tf.reshape(
                            predictions[layer_id][:, :, :, channel_id], new_shape
                        )
                    )[-nb_images_per_filter:]
                    max_activations = self._update_max_activations(
                        self.activations_model,
                        predictions,
                        layer_name,
                        image_batch,
                        max_activations,
                        batch_max_activations,
                        layer_id,
                        channel_id,
                        nb_images_per_filter,
                    )

        return max_activations

    @staticmethod
    def _check_dataset(img_dataset):
        """Check the provided dataset

        Raises
        ------
        TypeError
            Raise if the image dataset is not an iterable.
        TypeError
            Raise if the dataset images are not a numpy ndarray or a tensorflow Tensor.
        TypeError
            Raise if the shape of the dataset images is not as expected.
        """
        if not isinstance(img_dataset, Iterable):
            raise TypeError(
                "The image dataset must be an iterable, "
                "for instance like a tensorflow Dataset."
            )

        _, img_dataset_iter_copy = tee(iter(img_dataset))
        image_batch_example = next(img_dataset_iter_copy)
        if not isinstance(image_batch_example, (tf.Tensor, np.ndarray)):
            raise TypeError(
                "Dataset images must be numpy arrays or a tensorflow tensors."
            )

        if not len(image_batch_example.shape) == 4:
            raise TypeError(
                "Dataset images must yield batchs of images, with shape "
                "(batch size, height, width, number of channels). "
                "Shape of the first element of the provided dataset is "
                f"{image_batch_example.shape}."
            )

    def _locs_of_previous_layer(
        self,
        i: int,
        j: int,
        padding: str,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """Given the location of the activation of the current layer (i, j), the padding,
        the kernel size and the stride of the current layer, get the corresponding
        4 corners of the previous layer part that is used to compute the (i, j) value
        of the current layer.
        """
        if padding == "valid":
            shift = (0, 0)
        elif padding == "same":
            shift = tuple([-int(np.floor(x / 2)) for x in kernel_size])  # type: ignore
        else:
            raise ValueError(
                "The expected value for padding are 'valid' and 'same'. "
                f"The provided value {padding} is not handled."
            )
        return (
            i * strides[0] + shift[0],
            i * strides[0] + shift[0] + kernel_size[0] - 1,
            j * strides[1] + shift[1],
            j * strides[1] + shift[1] + kernel_size[1] - 1,
        )

    def _feature_map_loc_to_input_loc(
        self, model: Model, layer_name: str, i: int, j: int
    ) -> Tuple[int, int, int, int]:
        """Get the indices of the input image corresponding to the activation located
        at (i, j) in layer layer_name.
        """
        previous_layer = False
        x_min = i
        x_max = i
        y_min = j
        y_max = j
        for layer in reversed(model.layers):
            if layer.name == layer_name:
                previous_layer = True
            if previous_layer:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    strides = layer.strides
                    padding = layer.padding
                    kernel_size = layer.kernel_size
                    x_min, _, y_min, _ = self._locs_of_previous_layer(
                        x_min, y_min, padding, kernel_size, strides
                    )
                    _, x_max, _, y_max = self._locs_of_previous_layer(
                        x_max, y_max, padding, kernel_size, strides
                    )
                elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                    strides = layer.strides
                    padding = layer.padding
                    kernel_size = layer.pool_size
                    x_min, _, y_min, _ = self._locs_of_previous_layer(
                        x_min, y_min, padding, kernel_size, strides
                    )
                    _, x_max, _, y_max = self._locs_of_previous_layer(
                        x_max, y_max, padding, kernel_size, strides
                    )
        return x_min, x_max, y_min, y_max

    def get_filter_max_activations_from_image_dataset_visualization(
        self, max_activations: dict, **kwargs
    ) -> List[List[io.BytesIO]]:
        """For all the filters of the CNN, create BytesIO containing nb_images_per_filter
        of input image parts that maximize the activation of the filter.

        Parameters
        ----------
        max_activations : dict
            Results of get_filter_max_activations.
        **kwargs:
            kwargs for imshow, in particular for the color map.

        Returns
        -------
        List[List[io.BytesIO]]
            List of list of BytesIO.

        The results may be saved as png with the following command:
        desired_io = all_layer_io[3][0]
        desired_io.seek(0)
        with open("/path/to/image.png", "wb") as f:
            f.write(desired_io.read())
        """
        nb_images_per_filter = len(max_activations[0][0]["activation"])
        fig_subplot_nrows = int(np.ceil(np.sqrt(nb_images_per_filter)))  # type: ignore
        fig_subplot_ncols = fig_subplot_nrows

        all_layer_io = []
        for layer_id in max_activations.keys():
            layer_io = []
            nb_channels = len(max_activations[layer_id])
            for channel_id in range(nb_channels):
                size = max_activations[layer_id][channel_id]["input"][0].shape[0]
                fig = plt.figure(figsize=(size / 5, size / 5))
                for index in range(len(max_activations[layer_id][channel_id]["input"])):
                    ax = fig.add_subplot(
                        fig_subplot_nrows,
                        fig_subplot_ncols,
                        index + 1,
                    )
                    ax.imshow(
                        max_activations[layer_id][channel_id]["input"][index], **kwargs
                    )
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                b = io.BytesIO()
                plt.savefig(b, format="png")
                layer_io.append(b)
            all_layer_io.append(layer_io)
        return all_layer_io


if __name__ == "__main__":
    import os

    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    # Get model
    model = tf.keras.models.load_model(
        "/home/arnaudcapitaine/01_Projets/2021/demonstrator_image/data/model_cnn_v13"
    )

    # Get img_dataset
    mypath = "/home/arnaudcapitaine/01_Projets/2022/readml/data/images/"
    onlyfiles = [
        f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))
    ]
    batch = []
    for file in onlyfiles:
        image = load_img(mypath + file, grayscale=True, target_size=(200, 200))
        img_tensor = img_to_array(image)
        batch.append(np.expand_dims(img_tensor, 0))
    batch = np.concatenate(batch)
    img_dataset = tf.data.Dataset.from_tensor_slices(batch)
    img_dataset = img_dataset.batch(3)

    # test
    max_activation = MaxActivation(model)
    filter_max_activations = (
        max_activation.get_filter_max_activations_from_image_dataset(img_dataset)
    )
    all_layer_io = (
        max_activation.get_filter_max_activations_from_image_dataset_visualization(
            filter_max_activations, cmap="gray"
        )
    )
    desired_io = all_layer_io[3][0]
    desired_io.seek(0)
    with open("/home/arnaudcapitaine/Bureau/image.png", "wb") as f:
        f.write(desired_io.read())
