import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO


def file_to_grayscale(uploaded_file, zoom=2.0):
    """
    Convert uploaded file (image or PDF) to OpenCV grayscale image.

    Args:
        uploaded_file: Streamlit uploaded file object
        zoom: PDF rendering zoom factor (default: 2.0)

    Returns:
        numpy.ndarray: Grayscale image in OpenCV format
    """
    file_type = uploaded_file.type.lower()

    if file_type == 'application/pdf':
        return _pdf_to_grayscale(uploaded_file, zoom)
    else:
        return _image_to_grayscale(uploaded_file)


def _pdf_to_grayscale(uploaded_file, zoom):
    """Convert PDF first page to grayscale."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page = doc.load_page(0)  # First page only

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img_data = pix.tobytes("ppm")
    img = Image.open(BytesIO(img_data))
    doc.close()

    return _pil_to_opencv_gray(img)


def _image_to_grayscale(uploaded_file):
    """Convert image file to grayscale."""
    img = Image.open(uploaded_file)
    return _pil_to_opencv_gray(img)


def _pil_to_opencv_gray(pil_img):
    """Convert PIL image to OpenCV grayscale format."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    return opencv_img

# Usage example:
# if template:
#     gray_image = file_to_grayscale(template)
#     st.image(gray_image, caption="Processed Image", use_column_width=True)
