import cv2
import fitz
import numpy as np

from document_extraction.distortion import DocumentDistorter


def pdf_stream_2_img(pdf_bytes, zoom: int = 3):
    """    Convert PDF bytes to image bytes    """
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        # Only get first page
        page = pdf_doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return img_bytes

    finally:
        # Clean up
        pdf_doc.close()


def pdf_stream_2_cv2_gray(pdf_bytes, zoom: int = 3):
    """Convert PDF bytes directly to cv2 grayscale image"""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        page = pdf_doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        # Get grayscale pixmap directly - no PNG encoding/decoding
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

        # Convert to numpy array (cv2 format)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        gray_img = img_data.reshape(pix.height, pix.width)
        return gray_img

    finally:
        pdf_doc.close()


def pdf_stream_2_cv2_bgr(pdf_bytes, zoom: int = 3):
    """Convert PDF bytes directly to cv2 BGR image"""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        page = pdf_doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        # Get RGB pixmap (default colorspace)
        pix = page.get_pixmap(matrix=mat)

        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        rgb_img = img_data.reshape(pix.height, pix.width, 3)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return bgr_img

    finally:
        pdf_doc.close()


def process_distortion(distorted_pdf, **params):
    """Distortion"""
    distorted = pdf_stream_2_cv2_bgr(distorted_pdf)
    distorter = DocumentDistorter()
    distorted = distorter.apply_perspective_distortion(distorted, severity=params['severity'])
    distorted = distorter.apply_rotation(distorted, angle=params['angle'])
    distorted = distorter.apply_lighting_variation(distorted, max_shadow=params['max_shadow'])
    distorted = distorter.apply_noise(distorted, amount=params['amount'])

    return distorted
