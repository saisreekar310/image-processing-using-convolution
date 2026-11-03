"""Interactive Streamlit demonstration of convolution and image filters."""

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from convolution import inner_product
from filters import FILTER_DESCRIPTIONS, FILTER_KERNELS
from image_utils import convolve_image, load_image


PROJECT_DIR = Path(__file__).parent
SAMPLE_IMAGE_DIR = PROJECT_DIR / "sample_images"
EDGE_FILTERS = {"Sobel X", "Sobel Y", "Laplacian"}


def _show_kernel(kernel: np.ndarray) -> None:
    """Draw a labeled heatmap of the selected filter kernel."""
    figure, axis = plt.subplots()
    image = axis.imshow(kernel, cmap="coolwarm")
    axis.set_title("Filter kernel")
    axis.set_xticks(range(kernel.shape[1]))
    axis.set_yticks(range(kernel.shape[0]))
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            axis.text(
                column,
                row,
                f"{kernel[row, column]:g}",
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    st.pyplot(figure)
    plt.close(figure)


def _prepare_for_display(output: np.ndarray, filter_name: str) -> np.ndarray:
    """Convert convolution values to uint8 while keeping edge signs visible."""
    if filter_name in EDGE_FILTERS:
        output = np.abs(output)
        largest_value = float(np.max(output))
        if largest_value > 0:
            output = output * (255.0 / largest_value)
    return np.clip(output, 0, 255).astype(np.uint8)


def _load_selected_image(
    sample_paths: List[Path],
) -> Tuple[Optional[np.ndarray], str]:
    """Show image source controls and return the selected image, if any."""
    if sample_paths:
        source = st.radio("Image source", ("Upload an image", "Choose a sample"))
    else:
        source = "Upload an image"
        st.info("No sample images are available yet. Upload a JPG or PNG image.")

    if source == "Choose a sample":
        selected_path = st.selectbox(
            "Sample image",
            sample_paths,
            format_func=lambda path: path.name,
        )
        try:
            return load_image(selected_path), str(selected_path)
        except ValueError as error:
            st.error(str(error))
            return None, ""

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=("jpg", "jpeg", "png"),
        help="Choose a JPG, JPEG, or PNG image.",
    )
    if uploaded_file is None:
        return None, ""

    try:
        image = load_image(BytesIO(uploaded_file.getvalue()))
    except ValueError as error:
        st.error(str(error))
        return None, ""
    return image, uploaded_file.name


def _show_math_demonstration() -> None:
    """Present the convolution formula and a concrete inner-product example."""
    st.header("The mathematics")
    st.latex(
        r"(I * K)[i,j] = \sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1}"
        r"I[i+m,j+n]K[k_h-1-m,k_w-1-n]"
    )
    st.write(
        "The inner product of two equally sized arrays multiplies matching "
        "entries and adds the products. Convolution repeats that calculation "
        "for each image neighborhood. Here the kernel is flipped before it "
        "slides, and one zero-valued border pixel is added to preserve the "
        "image dimensions."
    )
    example_region = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]], dtype=float)
    example_kernel = FILTER_KERNELS["Laplacian"]
    example_value = inner_product(example_region, example_kernel)
    example_column, result_column = st.columns(2)
    with example_column:
        st.write("Example image region")
        st.dataframe(example_region, hide_index=True)
        st.write("Example kernel")
        st.dataframe(example_kernel, hide_index=True)
    with result_column:
        st.write("Element-wise products")
        st.dataframe(example_region * example_kernel, hide_index=True)
        st.metric("Inner product (sum of products)", f"{example_value:g}")
    st.write(
        "Blur kernels average nearby values, sharpening kernels emphasize "
        "local contrast, and edge kernels respond strongly where neighboring "
        "pixel intensities change."
    )


def main() -> None:
    st.set_page_config(page_title="Convolution and Image Processing", layout="wide")
    st.title("Convolution, Inner Product, and Image Processing Revisited")
    st.write(
        "Explore how a small matrix called a **kernel** interacts with an image. "
        "Each output pixel comes from an inner product between the kernel and "
        "a small image neighborhood."
    )

    _show_math_demonstration()
    st.header("1. Load an image")
    sample_paths = sorted(
        path
        for path in SAMPLE_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if SAMPLE_IMAGE_DIR.exists() else []
    image, source_name = _load_selected_image(sample_paths)
    if image is None:
        st.stop()

    st.caption(f"Loaded: {source_name} · image size: {image.shape[1]} × {image.shape[0]}")

    st.header("2. Choose a filter")
    filter_name = st.selectbox("Filter", list(FILTER_KERNELS))
    kernel = FILTER_KERNELS[filter_name]
    st.write(FILTER_DESCRIPTIONS[filter_name])
    _show_kernel(kernel)

    st.header("3. Process and compare")
    image_key = (source_name, image.shape, hash(image.tobytes()))
    if st.button("Process image", type="primary"):
        convolution_output = convolve_image(
            image,
            kernel,
            grayscale=filter_name in EDGE_FILTERS,
            padding=1,
            stride=1,
        )
        st.session_state["convolution_output"] = convolution_output
        st.session_state["processed_image_key"] = image_key
        st.session_state["processed_filter_name"] = filter_name

    has_current_result = (
        st.session_state.get("processed_image_key") == image_key
        and st.session_state.get("processed_filter_name") == filter_name
    )
    if has_current_result:
        convolution_output = st.session_state["convolution_output"]
        processed_image = _prepare_for_display(convolution_output, filter_name)
        output_color_mode = "RGB" if processed_image.ndim == 3 else "grayscale"
        original_column, processed_column = st.columns(2)
        with original_column:
            original_color_mode = "RGB" if image.ndim == 3 else "grayscale"
            st.image(
                image,
                caption=f"Original image ({original_color_mode})",
                width="stretch",
            )
        with processed_column:
            st.image(
                processed_image,
                caption=f"{filter_name} result ({output_color_mode})",
                width="stretch",
                clamp=True,
            )
        st.caption(
            f"Convolution output dimensions: {convolution_output.shape[0]} × "
            f"{convolution_output.shape[1]}"
        )
        with st.expander("View the first 10 × 10 output values"):
            if convolution_output.ndim == 3:
                red_tab, green_tab, blue_tab = st.tabs(("Red", "Green", "Blue"))
                for channel_tab, channel in zip(
                    (red_tab, green_tab, blue_tab), range(3)
                ):
                    with channel_tab:
                        st.dataframe(
                            convolution_output[:10, :10, channel],
                            width="stretch",
                        )
            else:
                st.dataframe(convolution_output[:10, :10], width="stretch")
    else:
        st.image(image, caption="Original image", width="stretch")
        st.info("Choose a filter and press **Process image** to see the result.")

if __name__ == "__main__":
    main()