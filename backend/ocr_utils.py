import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import os
import logging
from typing import Optional, Union
import io
import tempfile
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set  Tesseract path
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH


class OCRProcessor:
    def __init__(self, tesseract_path: Optional[str] = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f" Tesseract OCR is ready (version {version})")
        except EnvironmentError:
            logger.warning("️ Tesseract OCR not found. OCR will not work for scanned PDFs/images.")

    def extract_text_from_pdf(self, path: str, dpi: int = 300, fallback_ocr: bool = True) -> str:
        try:
            text = ""
            doc = fitz.open(path)  #  Works only with PyMuPDF installed

            for page_num, page in enumerate(doc):
                page_text = page.get_text().strip()

                if not page_text and fallback_ocr:
                    logger.info(f" Using OCR for page {page_num + 1}")
                    page_text = self._ocr_pdf_page(page, dpi)

                text += f"--- PAGE {page_num + 1} ---\n{page_text}\n\n"

            return text.strip()
        except Exception as e:
            logger.error(f" PDF processing failed: {str(e)}")
            raise ValueError(f"Could not process PDF: {str(e)}")

    def _ocr_pdf_page(self, page, dpi: int = 300) -> str:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, lang='eng')
        except Exception as e:
            logger.warning(f"Fallback OCR error: {str(e)}")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(
                        page.parent.name,
                        first_page=page.number + 1,
                        last_page=page.number + 1,
                        dpi=dpi,
                        output_folder=temp_dir
                    )
                    if images:
                        return pytesseract.image_to_string(images[0], lang='eng')
            except Exception as ex:
                logger.error(f"OCR fallback failed: {str(ex)}")
            return ""

    def extract_text_from_image(self, path: str, lang: str = 'eng') -> str:
        try:
            with Image.open(path) as img:
                img = self._preprocess_image(img)
                return pytesseract.image_to_string(img, lang=lang)
        except Exception as e:
            logger.error(f" Image processing failed: {str(e)}")
            raise ValueError(f"Could not process image: {str(e)}")

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        try:
            return img.convert('L') if img.mode != 'L' else img
        except:
            return img

    def extract_text(self, file_path: str, file_obj: Optional[Union[bytes, io.BytesIO]] = None) -> str:
        try:
            ext = os.path.splitext(file_path)[-1].lower()

            if ext == ".pdf":
                if file_obj:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                        temp_pdf.write(file_obj.read() if hasattr(file_obj, 'read') else file_obj)
                        temp_path = temp_pdf.name
                    try:
                        return self.extract_text_from_pdf(temp_path)
                    finally:
                        os.unlink(temp_path)
                return self.extract_text_from_pdf(file_path)

            elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                if file_obj:
                    with Image.open(file_obj) as img:
                        return pytesseract.image_to_string(img, lang='eng')
                return self.extract_text_from_image(file_path)

            elif ext == ".txt":
                if file_obj:
                    return file_obj.read().decode('utf-8') if hasattr(file_obj, 'read') else file_obj.decode('utf-8')
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except Exception as e:
            logger.error(f" Text extraction failed: {str(e)}")
            raise ValueError(f"Text extraction failed: {str(e)}")


#  Global instance
ocr_processor = OCRProcessor()
