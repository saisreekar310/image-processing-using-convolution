"""Small, explicit 2D convolution routines for learning image processing."""

import numpy as np


def inner_product(region: np.ndarray, kernel: np.ndarray) -> float:
    """Return the sum of element-wise products for equally sized 2D arrays."""
    region_array = np.asarray(region)
    kernel_array = np.asarray(kernel)

    if region_array.ndim != 2 or kernel_array.ndim != 2:
        raise ValueError("The region and kernel must both be 2D arrays.")
    if region_array.shape != kernel_array.shape:
        raise ValueError("The region and kernel must have the same shape.")

    return float(np.sum(region_array * kernel_array))


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """Apply a 2D convolution with zero padding and a positive stride.

    Convolution flips the kernel horizontally and vertically before sliding it
    over the image. Each output pixel is the inner product of one image region
    and the flipped kernel.
    """
    image_array = np.asarray(image)
    kernel_array = np.asarray(kernel)

    if image_array.ndim != 2 or kernel_array.ndim != 2:
        raise ValueError("Image and kernel must both be 2D arrays.")
    if image_array.size == 0 or kernel_array.size == 0:
        raise ValueError("Image and kernel must not be empty.")
    if not np.issubdtype(image_array.dtype, np.number):
        raise TypeError("Image values must be numeric.")
    if not np.issubdtype(kernel_array.dtype, np.number):
        raise TypeError("Kernel values must be numeric.")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("Padding must be a non-negative integer.")
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("Stride must be a positive integer.")

    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel_array.shape
    padded_height = image_height + 2 * padding
    padded_width = image_width + 2 * padding

    if kernel_height > padded_height or kernel_width > padded_width:
        raise ValueError("Kernel must fit inside the padded image.")

    padded_image = np.pad(
        image_array.astype(float),
        ((padding, padding), (padding, padding)),
        mode="constant",
    )
    flipped_kernel = np.flip(kernel_array.astype(float), axis=(0, 1))
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width), dtype=float)

    for output_row in range(output_height):
        row_start = output_row * stride
        for output_column in range(output_width):
            column_start = output_column * stride
            image_region = padded_image[
                row_start : row_start + kernel_height,
                column_start : column_start + kernel_width,
            ]
            output[output_row, output_column] = inner_product(
                image_region, flipped_kernel
            )

    return output