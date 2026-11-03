# Convolution, Inner Product, and Image Processing Revisited

A beginner-friendly Python project that demonstrates how inner products and
2D convolution can be used to process digital images. The Streamlit app lets
you upload a JPG/JPEG/PNG image, choose a filter, inspect its kernel, and
compare the input with the filtered result.

## Problem statement

Digital images are arrays of pixel values. Many useful image effects can be
created by moving a small matrix (a **kernel** or **filter**) over an image.
At each position, corresponding values are multiplied and added. This is an
inner product, and repeating it over all positions is 2D convolution.

The project implements this operation directly with NumPy loops so the
underlying mathematics remains visible instead of being hidden inside a
prebuilt image-filtering function.

## Project objectives

- Load and preview JPG, JPEG, and PNG images.
- Preserve RGB color for identity, blur, and sharpening filters.
- Convert RGB images to grayscale for edge-detection filters.
- Explain inner products with a small numerical example.
- Implement 2D convolution with zero padding and stride.
- Compare seven common image-processing kernels.
- Show the selected kernel, processed image, and a small numeric output preview.
- Test the math and image-loading helpers with small inputs.

## Project structure

```text
image_processing_project/
├── app.py                    # Streamlit interface and math demonstration
├── convolution.py            # Inner product and manual 2D convolution
├── filters.py                # Filter kernels and descriptions
├── image_utils.py            # Image loading and grayscale conversion
├── requirements.txt          # Python dependencies
├── README.md
├── sample_images/            # Optional local images for the app's sample picker
└── tests/
    ├── test_convolution.py
    └── test_image_utils.py
```

## Mathematical concepts

### Inner product

For two vectors of equal length, their inner product is

\[
\langle \mathbf{u}, \mathbf{v} \rangle
= \sum_{i=1}^{n} u_i v_i.
\]

For matrices of equal shape, the same idea is applied to matching positions:
multiply each pair of entries, then add all the products. In the program,
`inner_product(region, kernel)` performs this calculation.

Example:

```text
Image region                 Kernel
┌       ┐                     ┌       ┐
│ 1 2 0 │                     │ 0  1 0│
│ 0 1 3 │         and         │ 1 -4 1│
│ 2 1 0 │                     │ 0  1 0│
└       ┘                     └       ┘
```

The element-wise products are

```text
0  2  0
0 -4  3
0  1  0
```

Their sum is \(0+2+0+0-4+3+0+1+0=2\). That single number is the result for
this image neighborhood and kernel.

### 2D convolution

Convolution slides a kernel over an image. At each position, it multiplies
the image neighborhood and kernel element by element and sums the result:

\[
Y[i,j] =
\sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1}
X_p[iS+m,\ jS+n]\,
K[k_h-1-m,\ k_w-1-n].
\]

Here \(X_p\) is the padded image, \(K\) is the kernel, \(k_h\) and \(k_w\)
are the kernel height and width, and \(S\) is the stride. The reversed kernel
indices show that `convolve2d` implements mathematical convolution by flipping
the kernel horizontally and vertically before applying it. Many image APIs
instead perform cross-correlation without flipping; symmetric kernels produce
the same result either way.

Zero padding adds a border of zero-valued pixels around the image. For an
image of height \(H\) and width \(W\), kernel dimensions \(k_h \times k_w\),
padding \(P\), and stride \(S\), the output dimensions are

\[
H_{out} = \left\lfloor \frac{H+2P-k_h}{S}\right\rfloor+1,
\qquad
W_{out} = \left\lfloor \frac{W+2P-k_w}{S}\right\rfloor+1.
\]

The app uses padding 1 and stride 1 for its 3-by-3 filters, preserving the
input image dimensions.

## Filter kernels

The matrices below are the exact kernels defined in `filters.py`.

### Identity

\[
\begin{bmatrix}
0&0&0\\
0&1&0\\
0&0&0
\end{bmatrix}
\]

Keeps the center pixel unchanged.

### Box blur

\[
\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}
\]

Averages the 3-by-3 neighborhood to soften details.

### Gaussian blur

\[
\frac{1}{16}
\begin{bmatrix}
1&2&1\\
2&4&2\\
1&2&1
\end{bmatrix}
\]

Gives more weight to pixels near the center for a smoother blur.

### Sharpen

\[
\begin{bmatrix}
0&-1&0\\
-1&5&-1\\
0&-1&0
\end{bmatrix}
\]

Emphasizes local contrast around each pixel.

### Sobel X

\[
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}
\]

Responds to intensity changes across columns and highlights vertical edges.

### Sobel Y

\[
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}
\]

Responds to intensity changes across rows and highlights horizontal edges.

### Laplacian

\[
\begin{bmatrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{bmatrix}
\]

Responds to rapid intensity changes in any direction.

## Algorithms and image display

1. Pillow reads the uploaded file and checks that its actual format is JPEG or
   PNG. Images are converted to RGB for the original-image preview.
2. Identity, box blur, Gaussian blur, and sharpen apply `convolve2d`
   independently to the red, green, and blue channels. Sobel X, Sobel Y, and
   Laplacian first convert RGB pixels to grayscale using
   \(0.299R + 0.587G + 0.114B\), then apply the kernel.
3. `convolve2d` flips the kernel, pads the image, then uses nested loops to
   extract each neighborhood and call the inner-product operation. Grayscale
   image input is also supported.
4. The result is displayed next to the original and labeled RGB or grayscale.
   Display values are clipped to 0–255. For edge filters, absolute response
   values are scaled for visibility; the numeric convolution output is
   unchanged.
5. Matplotlib draws a labeled heatmap of the selected kernel. The app also
   shows up to the first 10-by-10 values from the numeric convolution output,
   with separate channel tabs for RGB results.

## Installation

Open PowerShell in the project directory and run:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, use the interpreter
directly instead:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

From the project directory, with the virtual environment active:

```powershell
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually
`http://localhost:8501`). Alternatively, run it without activating the
environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Choose a JPG, JPEG, or PNG file in the upload section, select a filter, and
press **Process image**. If you add supported images to `sample_images/`, they
will also be available in the local sample picker.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover inner-product arithmetic, a small convolution example,
identity filtering with padding, output dimensions with different padding
and stride values, invalid inputs, image format validation, and grayscale
conversion.

## Expected output

- The app displays an original image and the chosen filter kernel.
- After processing, color-preserving filters show an RGB result, while edge
  filters show a grayscale result beside the original.
- The displayed output retains the original dimensions for the app's 3-by-3
  filters.
- The numeric output preview and the mathematical example make it possible to
  connect a visible pixel effect to the inner-product calculation.
- Invalid or unreadable image data produces a clear error message.

## Conclusion and future improvements

This project connects a foundational linear-algebra operation—the inner
product—to convolution and practical image filters. Because the core operation
is implemented with explicit loops, it is suitable for studying how the
kernel and each image neighborhood combine.

Possible extensions include adjustable kernel size and stride in the
interface, additional filters, and side-by-side comparison of several filter
outputs.