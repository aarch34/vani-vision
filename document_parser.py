"""
document_parser.py - Extract text from PDF and Word (DOCX) files.
"""

import io
from loguru import logger

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[PDF parsing unavailable - Please install PyMuPDF]"
    
    text = ""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf_document:
            text += page.get_text() + "\n"
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return f"[Error parsing PDF: {e}]"
    return text.strip()

def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    try:
        import docx
    except ImportError:
        return "[DOCX parsing unavailable - Please install python-docx]"
        
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error parsing DOCX: {e}")
        return f"[Error parsing DOCX: {e}]"
    return text.strip()

def parse_document(file_bytes: bytes, file_name: str) -> str:
    """Determine file type and extract text."""
    lower_name = file_name.lower()
    if lower_name.endswith('.pdf'):
        return parse_pdf(file_bytes)
    elif lower_name.endswith('.docx'):
        return parse_docx(file_bytes)
    else:
        logger.error(f"Unsupported document format: {file_name}")
        return ""
