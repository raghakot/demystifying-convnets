import cv2
import os
import tempfile
import math

import numpy as np
from skimage import io
from keras.models import load_model


def load_img(path, grayscale=False, target_size=None):
    """Utility function to load an image from disk.

    Args:
      path: The image file path.
      grayscale: True to convert to grayscale image (Default value = False)
      target_size: (w, h) to resize. (Default value = None)

    Returns:
        The loaded numpy image.
    """
    img = io.imread(path, grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img


def stitch_images(images, margin=5, cols=5):
    """Utility function to stitch images together with a `margin`.

    Args:
        images: The array of images (batch, height, width) to stitch.
        margin: The black border margin size between images (Default value = 5)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)

    Returns:
        A single numpy image array comprising of input images.
    """
    n, w, h, = images.shape
    n_rows = max(1, int(math.ceil(n / cols)))
    n_cols = min(n, cols)

    out_w = n_cols * w + (n_cols - 1) * margin
    out_h = n_rows * h + (n_rows - 1) * margin
    stitched_images = np.zeros((out_h, out_w), dtype=images.dtype)

    for row in range(n_rows):
        for col in range(n_cols):
            img_idx = row * cols + col
            if img_idx >= n:
                break

            stitched_images[(h + margin) * row : (h + margin) * row + h,
            (w + margin) * col : (w + margin) * col + w] = images[img_idx]

    return stitched_images


def apply_modifications(model):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.

    Args:
        model: The `keras.models.Model` instance.

    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path)
    finally:
        os.remove(model_path)
    from keras.activations import linear
