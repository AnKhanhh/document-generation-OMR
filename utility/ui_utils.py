import fitz


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
