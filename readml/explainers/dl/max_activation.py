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
            max_activations[self.conv2d_layers[layer_id][0]] = {}
            for channel_id in range(activations_model.output[layer_id].shape[-1]):
                max_activations[self.conv2d_layers[layer_id][0]][channel_id] = {
                    "activation": [],
                    "input": [],
                }
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
                overall_max_activations[layer_name][channel_id]["activation"]
            ) == nb_images_per_filter and batch_max_activation.numpy() > min(
                overall_max_activations[layer_name][channel_id]["activation"]
            ):
                index_to_remove = overall_max_activations[layer_name][channel_id][
                    "activation"
                ].index(
                    min(overall_max_activations[layer_name][channel_id]["activation"])
                )
                del overall_max_activations[layer_name][channel_id]["activation"][
                    index_to_remove
                ]
                del overall_max_activations[layer_name][channel_id]["input"][
                    index_to_remove
                ]

            # if there are less than nb_images_per_filter in overall_max_activations
            # then add batch_max_activation
            if (
                len(overall_max_activations[layer_name][channel_id]["activation"])
                < nb_images_per_filter
            ):
                overall_max_activations[layer_name][channel_id]["activation"].append(
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
                overall_max_activations[layer_name][channel_id]["input"].append(
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
            Results of get_filter_max_activations_from_image_dataset.
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
        nb_images_per_filter = len(next(iter(max_activations.items()))[1])
        fig_subplot_nrows = int(np.ceil(np.sqrt(nb_images_per_filter))) # type: ignore
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
                        max_activations[layer_id][channel_id]["input"][index],
                        **kwargs,
                    )
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                b = io.BytesIO()
                plt.savefig(b, format="png")
                layer_io.append(b)
            all_layer_io.append(layer_io)
        return all_layer_io

    def _build_submodel(
        self,
        conv_layer_name: str,
        add_input_over_all_reals: bool,
        input_range: List[float],
        change_relu: bool,
    ) -> Model:
        """
        Build a submodel of the original model.

        The output of the submodel is the output of the Conv2D layer which name is provided in argument.

        Since the image pixel values are between 0 and 1 (after normalization),
        an additional layer may be added at the beginning implementing a sigmoid
        to map all the real to values between 0 and 1.

        The derivative of the ReLU is 0 for negative values. This may prevent the optimization to work.
        It is possible to change the ReLU activation by a Leaky ReLU one in order to make the optimization works.
        """
        # Create a copy because activations may be changed
        model_copy = tf.keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

        # Create submodel
        submodel = Model(
            model_copy.input,
            [model_copy.get_layer(conv_layer_name).output, model_copy.input],
        )

        # Change activation
        if change_relu:
            layer_with_activation_list = [
                layer for layer in submodel.layers if hasattr(layer, "activation")
            ]
            for layer in layer_with_activation_list:
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = tf.nn.leaky_relu

        # Add new input over all reals in order to restrain the image value over ]0,1[
        if add_input_over_all_reals:
            if not len(input_range) == 2:
                raise ValueError("The provided input_range must be a list of 2 values.")
            input_layer_over_all_reals = tf.keras.Input(
                shape=submodel.inputs[0].shape[-3:]
            )
            x = tf.keras.layers.Activation(
                lambda x: (input_range[1] - input_range[0])
                * tf.keras.activations.sigmoid(x)
                + input_range[0]
            )(input_layer_over_all_reals)
            output = submodel(x)[0]
            submodel = Model(input_layer_over_all_reals, [output, x])

        return submodel

    def create_input_image_maximizing_activations(
        self,
        layer_name: str,
        channel_id: int,
        add_input_over_all_reals: bool,
        input_range: List[float],
        max_it: int,
        change_relu: bool = True,
        nb_activation_to_max: str = "single",
        learning_rate: float = 1.0,
        grad_norm_eps: float = 1e-6,
    ) -> Tuple[tf.Tensor, list, list]:
        """
        Create an input image that maximize the output of the provided layer and channel.

        Parameters
        ----------
        layer_name : str
            layer name containing the filter to consider
        channel_id : int
            channel id of the filter to consider
        add_input_over_all_reals : bool
            Whether to add a layer before the input that is unconstrainted.
            It is linked to the first layer by means of a sigmoid activation,
            so that all reals are mapped to restricted range of value
            ] input_range[0] , input_range[1] [.
            It is useful when performing the optimization since an unconstrained
            optimization algorithm (gradient ascent) is used to get the input image
            that maximize the activation.
        input_range : List[float]
            Range of value of the input image.
            Generally, it is [0, 1] or [0, 255].
        max_it : int
            Number max of iteration.
        change_relu : bool, optional
            Whether to change the relu activation to weaky relu activation.
            Relu may prevent the optimization to progress since the gradient of relu
            is either 1 or 0. The algorithm may be stuck.
            By default True
        nb_activation_to_max : str, optional
            Two approaches are implemented, by default "single".
            - "single" optimize a single activation pixel.
            Consequently, only a part of the input image is retrieved at the end.
            - "all" optimize the mean of all the activation pixels.
            Consequently, an entire input image is retrieved at the end.
        learning_rate : float, optional
            Learning rate of the gradient ascent, by default 1.0.
        grad_norm_eps : float, optional
            Stop condition over the gradient norm, by default 1e-6.

        Returns
        -------
        Tuple[tf.Tensor, list, list]
            - Input image that maximize the activation of the desired layer and channel.
            It is either a part of the image if nb_activation_to_max == "single" or
            the entire image if nb_activation_to_max == "all".
            - History of the mean activations. It should be increasing.
            - History of the gradient norm.

        Raises
        ------
        ValueError
            _description_
        """
        submodel = self._build_submodel(
            layer_name, add_input_over_all_reals, input_range, change_relu
        )
        # Initialize image value
        image_size = (1, *submodel.inputs[0].shape[-3:])
        if add_input_over_all_reals:
            input_data = np.random.normal(size=image_size)
        else:
            input_data = np.random.uniform(
                low=input_range[0], high=input_range[1], size=image_size
            )
        # Cast random noise from np.float64 to tf.float32 Variable because we will compute the derivative
        input_data = tf.Variable(tf.cast(input_data, tf.float32))

        # get maximum size to look at during optimization
        if nb_activation_to_max == "single":
            feature_map_x = 0
            feature_map_y = 0
            x_min, x_max, y_min, y_max = self._feature_map_loc_to_input_loc(
                self.model, layer_name, feature_map_x, feature_map_y
            )
            if min(x_min, y_min) < 0:
                feature_map_x -= x_min
                feature_map_y -= y_min
                x_min, x_max, y_min, y_max = self._feature_map_loc_to_input_loc(
                    self.model, layer_name, feature_map_x, feature_map_y
                )
            assert (
                x_min >= 0
                and y_min >= 0
                and x_max < submodel.inputs[0].shape[1]
                and y_max < submodel.inputs[0].shape[2]
            )
        elif nb_activation_to_max == "all":
            x_min, y_min = 0, 0
            x_max, y_max = input_data.shape[1:3]
        else:
            raise ValueError(f"Unknown input argument {nb_activation_to_max}.")

        # Iterate gradient ascents
        mean_activation_history = []
        grad_history = []
        for _ in range(max_it):
            with tf.GradientTape() as tape:
                output = submodel(input_data)[0]
                if nb_activation_to_max == "single":
                    mean_activation = tf.reduce_mean(
                        output[:, feature_map_x, feature_map_y, channel_id]
                    )
                elif nb_activation_to_max == "all":
                    mean_activation = tf.reduce_mean(output[:, :, :, channel_id])
                mean_activation_history.append(mean_activation)
            grads = tape.gradient(mean_activation, input_data)
            if tf.norm(grads) < grad_norm_eps:
                print("Optimization stopped since gradient is almost flat.")
                break
            grad_history.append(tf.norm(grads))
            input_data.assign_add(grads * learning_rate)
        return (
            submodel(input_data)[1][0, x_min : x_max + 1, y_min : y_max + 1, :],
            mean_activation_history,
            grad_history,
        )
