import os
import logging
import hashlib
from typing import List, Dict, Any, Tuple
import tempfile
import pdfplumber
from docx import Document as DocxDocument
from email import message_from_string
from email.policy import default as email_policy
from models.schemas import DocumentType
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentIngestionService:
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.eml', '.txt'}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100"))
        )

    def process_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        try:
            file_extension = os.path.splitext(filename.lower())[1]
            if file_extension == '.pdf':
                text_content = self._extract_pdf_text(file_content)
                doc_type = DocumentType.PDF
            elif file_extension == '.docx':
                text_content = self._extract_docx_text(file_content)
                doc_type = DocumentType.DOCX
            else:
                 raise ValueError(f"Unsupported file type: {file_extension}")

            if not text_content.strip():
                raise ValueError("No text content could be extracted from the file")

            chunks = self.text_splitter.split_text(text_content)
            document_id = self._generate_document_id(filename, file_content)
            
            metadatas = []
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "document_type": doc_type.value,
                    "chunk_index": i
                }
                metadatas.append(metadata)
                chunk_ids.append(chunk_id)

            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created")
            return chunks, metadatas, chunk_ids
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {str(e)}")
            raise

    def _generate_document_id(self, filename: str, content: bytes) -> str:
        content_hash = hashlib.md5(content).hexdigest()[:8]
        clean_filename = os.path.splitext(filename)[0][:20]
        return f"{clean_filename}_{content_hash}"

    def _extract_pdf_text(self, file_content: bytes) -> str:
        temp_path = ""
        try:
            # Create a temporary file, but tell it NOT to delete when closed
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name # Get the path of the created file
            
            # At this point, the 'with' block is finished, so the file is closed and unlocked.
            # Now we can safely open it with pdfplumber using its path.
            text_content = []
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            return "\n\n".join(text_content)
        finally:
            # Manually clean up the temporary file after we are done
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _extract_docx_text(self, file_content: bytes) -> str:
        temp_path = ""
        try:
            # Apply the same robust pattern for DOCX files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            doc = DocxDocument(temp_path)
            return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def process_document_from_url(self, url: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        logger.info(f"Downloading PDF from URL: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            file_content = response.content
            filename = os.path.basename(url.split('?')[0])
            return self.process_uploaded_file(file_content, filename)
        except requests.RequestException as e:
            logger.error(f"Failed to download file from URL {url}: {e}")
            raise ValueError(f"Could not download file from URL: {e}") from e

ingestion_service = DocumentIngestionService()
