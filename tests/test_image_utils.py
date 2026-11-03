import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from image_utils import convolve_image, load_image, to_grayscale


class ImageLoadingTests(unittest.TestCase):
    def test_load_image_returns_rgb_array(self):
        image_buffer = BytesIO()
        Image.new("RGB", (2, 1), color=(255, 0, 0)).save(
            image_buffer, format="PNG"
        )
        image_buffer.seek(0)

        result = load_image(image_buffer)

        self.assertEqual(result.shape, (1, 2, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])

    def test_load_image_rejects_unsupported_format(self):
        image_buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(image_buffer, format="BMP")
        image_buffer.seek(0)

        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            load_image(image_buffer)

    def test_load_image_rejects_unreadable_data(self):
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            load_image(BytesIO(b"not an image"))


class GrayscaleTests(unittest.TestCase):
    def test_rgb_image_converts_using_luminance_weights(self):
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)

        result = to_grayscale(image)

        np.testing.assert_allclose(result, [[76.245]])

    def test_grayscale_input_is_returned_as_float(self):
        image = np.array([[10, 20]], dtype=np.uint8)

        result = to_grayscale(image)

        np.testing.assert_array_equal(result, [[10.0, 20.0]])
        self.assertEqual(result.dtype, float)

    def test_invalid_image_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((2, 2, 4)))


class ImageConvolutionTests(unittest.TestCase):
    def test_identity_filter_preserves_rgb_image(self):
        image = np.array(
            [
                [[255, 0, 0], [0, 200, 0]],
                [[0, 0, 100], [40, 80, 120]],
            ],
            dtype=np.uint8,
        )

        result = convolve_image(image, np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ))

        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(result, image)

    def test_color_blur_convolves_each_channel_independently(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[1, 1] = [90, 150, 210]
        box_blur = np.full((3, 3), 1 / 9)

        result = convolve_image(image, box_blur)

        self.assertEqual(result.shape, (3, 3, 3))
        self.assertGreater(result[1, 1, 0], 0)
        self.assertGreater(result[1, 1, 1], result[1, 1, 0])
        self.assertGreater(result[1, 1, 2], result[1, 1, 1])

    def test_edge_filter_converts_rgb_to_grayscale_before_convolution(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[:, 2, :] = [200, 100, 50]
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        result = convolve_image(image, sobel_x, grayscale=True)
        expected = convolve_image(to_grayscale(image), sobel_x)

        self.assertEqual(result.shape, (3, 3))
        np.testing.assert_allclose(result, expected)
        self.assertGreater(np.max(np.abs(result)), 0)

    def test_identity_filter_preserves_grayscale_input(self):
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        result = convolve_image(image, identity)

        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(result, image)


if __name__ == "__main__":
    unittest.main()
