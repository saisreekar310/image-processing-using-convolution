"""Image loading and simple color conversion helpers."""

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image

from convolution import convolve2d


ImageSource = Union[str, Path, BinaryIO]
SUPPORTED_FORMATS = {"JPEG", "PNG"}


def load_image(source: ImageSource) -> np.ndarray:
    """Load a JPG/JPEG or PNG image and return an RGB uint8 array."""
    try:
        with Image.open(source) as image:
            if image.format not in SUPPORTED_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_FORMATS))
                raise ValueError(f"Unsupported image format. Use: {supported}.")
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as error:
        raise ValueError("The uploaded file is not a readable image.") from error


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB image array to floating-point grayscale."""
    image_array = np.asarray(image)

    if not np.issubdtype(image_array.dtype, np.number):
        raise TypeError("Image values must be numeric.")
    if image_array.ndim == 2:
        return image_array.astype(float)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image must be a 2D grayscale or 3-channel RGB array.")

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def convolve_image(
    image: np.ndarray,
    kernel: np.ndarray,
    grayscale: bool = False,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Convolve grayscale input or independently process each RGB channel."""
    image_array = np.asarray(image)

    if grayscale:
        image_array = to_grayscale(image_array)
    elif image_array.ndim == 2:
        return convolve2d(image_array, kernel, padding=padding, stride=stride)
    elif image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image must be a 2D grayscale or 3-channel RGB array.")

    if image_array.ndim == 2:
        return convolve2d(image_array, kernel, padding=padding, stride=stride)

    return np.stack(
        [
            convolve2d(
                image_array[:, :, channel],
                kernel,
                padding=padding,
                stride=stride,
            )
            for channel in range(3)
        ],
        axis=2,
    )