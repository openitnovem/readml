from typing import Any, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class GradCAM:
    """
    Allows to explain a DL model predictions on image data.

    Attributes
    ----------
    model : TensorFlow or Keras model
        TensorFlow models and Keras models using the TensorFlow backend are supported.
    class_idx : int
        The class index used to measure the class activation map
    layer_name : str, Optional
        layer to be used when visualizing the class activation map.
        If None, we try to automatically find the target output layer
    """

    def __init__(
        self, model: Any, class_idx: int, layer_name: Optional[str] = None
    ) -> None:

        # Store the model, the class index used to measure the class activation map,
        # and the layer to be used when visualizing the class activation map
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        # if the layer name is None, attempt to automatically find the target output layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self) -> str:
        """
        Attempt to find the final convolutional layer in the network by looping over the
        layers of the network in reverse order.

        Returns
        -------
        str
            Final convolutional layer in the network
        """
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(
        self, image: np.ndarray, eps: Optional[float] = 1e-8
    ) -> np.ndarray:
        """
        Compute the GRAD-CAM heatmap

        Parameters
        ----------
        image : np.ndarray
            Preprocessed image to explain
        eps : float, optional
            Float add to the denominator while normalizing the heatmap to avoid ending
            with a denominator of 0.
            It doesn't affect the resulting heatmap values
            The default is 1e-8

        Returns
        -------
        np.ndarray
            GRAD-CAM heatmap (grayscale representation of where the network activated in
            the image)
        """
        # construct our gradient model by supplying (1) the inputs to our pre-trained model,
        # (2) the output of the (presumably) final 4D layer in the network, and (3) the output of
        # the softmax activations from the model
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
        )
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the image through the gradient
            # model, and grab the loss associated with the specific class index
            (conv_outputs, predictions) = grad_model(tf.cast(image, tf.float32))
            if predictions.shape[-1] == 1:
                loss = predictions[0]
            else:
                loss = predictions[:, self.class_idx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, conv_outputs) + eps
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")

        # compute the guided gradients
        guided_grads = cast_conv_outputs * cast_grads * grads

        # the convolution and guided gradients have a batch dimension (which we don't need)
        # so we will grab the volume itself and discard the batch
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        # compute the average of the gradient values, and using them as weights, compute
        # the ponderation of the filters with respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        # grab the spatial dimensions of the input image and resize the output class activation
        # map to match the input image dimensions
        (width, height) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (width, height))
        # normalize the heatmap such that all values lie in the range [0, 1], scale the resulting
        # values, and then convert to an unsigned 8-bit integer
        heatmap = (heatmap - np.min(heatmap)) / ((heatmap.max() - heatmap.min()) + eps)
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: Optional[float] = 0.5,
        colormap: Optional[int] = cv2.COLORMAP_PARULA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the supplied color map to the heatmap and then overlay the heatmap on the
        input image

        Parameters
        ----------
        heatmap : np.ndarray
            The resulting heatmap of Guided GRAD-CAM (grayscale representation of where
            the network activated in the image)
        image : np.ndarray
            Original image
        alpha : float, optional
            Alpha channel, used in transparent overlays
        colormap : int, optional
            Colormap to apply to the Guided GRAD-CAM heatmap
            Default is OpenCV’s built in PARULA colormap (cv2.COLORMAP_PARULA)

        Returns
        -------
        Tuple[np.ndarray,np.ndarray]
            2-tuple of the color mapped heatmap and the output, overlaid image
        """

        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return heatmap, output
