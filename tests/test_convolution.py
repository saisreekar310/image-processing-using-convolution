import unittest

import numpy as np

from convolution import convolve2d, inner_product
from filters import FILTER_KERNELS


class InnerProductTests(unittest.TestCase):
    def test_inner_product_sums_elementwise_products(self):
        region = np.array([[1, 2], [3, 4]])
        kernel = np.array([[2, 0], [1, 3]])

        self.assertEqual(inner_product(region, kernel), 17.0)

    def test_inner_product_rejects_different_shapes(self):
        with self.assertRaises(ValueError):
            inner_product(np.ones((2, 2)), np.ones((3, 3)))


class ConvolutionTests(unittest.TestCase):
    def test_convolution_matches_small_numerical_example(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        kernel = np.array([[1, 0], [0, -1]])

        result = convolve2d(image, kernel)

        np.testing.assert_array_equal(result, [[4, 4], [4, 4]])

    def test_identity_filter_preserves_image_with_padding(self):
        image = np.array([[1, 2], [3, 4]], dtype=float)

        result = convolve2d(image, FILTER_KERNELS["Identity"], padding=1)

        np.testing.assert_array_equal(result, image)

    def test_output_dimensions_follow_padding_and_stride(self):
        image = np.ones((5, 6))
        kernel = np.ones((3, 2))

        result = convolve2d(image, kernel, padding=1, stride=2)

        self.assertEqual(result.shape, (3, 4))

    def test_kernel_larger_than_padded_image_is_rejected(self):
        with self.assertRaises(ValueError):
            convolve2d(np.ones((2, 2)), np.ones((3, 3)))

    def test_invalid_stride_is_rejected(self):
        with self.assertRaises(ValueError):
            convolve2d(np.ones((3, 3)), np.ones((2, 2)), stride=0)


if __name__ == "__main__":
    unittest.main()