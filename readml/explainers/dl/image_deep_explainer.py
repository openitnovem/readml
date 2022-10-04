# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Optional, Tuple

import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from readml.explainers.dl.grad_cam import GradCAM
from readml.logger import logger


class VisualExplainer:
    """
    Allows to explain a DL model predictions on image data.

    Attributes
    ----------
    model : TensorFlow model
        TensorFlow models and Keras models using the TensorFlow backend are supported
        Model to compute predictions using provided data.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.color_mode = None

    def local_explainer(
        self,
        img_dir: str,
        path_out: str,
        size: Optional[Tuple] = (224, 224),
        color_mode: Optional[str] = "rgb",
    ):
        """
        Loops over image directory, computes and saves GRAD-CAM heatmaps for each image
        in the given output path

        Parameters
        ----------
        img_dir : str
            Path of the directory containing images to explain.
        path_out : str
            Output directory path, used to save interpretability images
        size : tuple, optional
            Tuple of ints (img_height, img_width)
            The default is (224, 224)
        color_mode : {"grayscale","rgb"}
            Choose "rgb" or "grayscale"
            The default is rgb
        Returns
        -------
        None
        """
        self.color_mode = color_mode
        # Loop over image directory
        list_images = [x for x in os.listdir(img_dir)]
        for img_path in list_images:
            root_image_path = os.path.join(img_dir, img_path)
            image, orig = self.get_img_array(
                root_image_path, size, color_mode=color_mode
            )
            # Start explainer use the network to make predictions on the input image and find
            # the class label index with the largest corresponding probability
            output = self.gradcam_explainer(image=image, orig=orig)
            output_path = os.path.join(path_out, "grad_cam_" + img_path)
            cv2.imwrite(
                img=output,
                filename=output_path,
            )
            logger.info(f"GRAD_CAM computed on {img_path} and saved in {output_path}")

    @staticmethod
    def get_img_array(
        img_path: str, size: tuple, color_mode: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the input image from disk (in Keras/TensorFlow format) and preprocess it

        Parameters
        ----------
        img_path : str
            Path of the image to explain.
        size : tuple
            Tuple of ints (img_height, img_width)
        color_mode : {"grayscale","rgb"}
            Choose "rgb" or "grayscale"

        Returns
        -------
        image, orig: Tuple[np.ndarray, np.ndarray]
            preprocessed image and origin image
        """
        if color_mode == "grayscale":
            orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            orig = cv2.resize(orig, (size[1], size[0]))

        else:
            orig = cv2.imread(img_path)
        # load the input image from disk (in Keras/TensorFlow format) and preprocess it
        image = load_img(img_path, target_size=size, color_mode=color_mode)
        image_array = img_to_array(image)
        image_expanded = np.expand_dims(image_array, axis=0)
        return image_expanded, orig

    def gradcam_explainer(self, image: np.ndarray, orig: np.ndarray) -> np.ndarray:
        """
        Compute GRAD-CAM on a given image and concat the original image and the
        resulting GRAD-CAM heatmap.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed image used to compute GRAD-CAM heatmap
        orig : np.ndarray
            Original image

        Returns
        -------
        output : np.ndarray
            output image tha contains the original image, GRAD-CAM heatmap and the
            overlay of the image and heatmap
        """
        preds = self.model.predict(image)
        i = np.argmax(preds[0])
        # initialize our guided gradient class activation map and build the heatmap
        cam = GradCAM(model=self.model, class_idx=i)
        heatmap = cam.compute_heatmap(image)
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        if self.color_mode == "grayscale":
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        heatmap, output = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
        # Concat the original image and resulting heatmap and output image
        output = np.vstack([orig, heatmap, output])
        output = imutils.resize(output, height=700)
        return output
