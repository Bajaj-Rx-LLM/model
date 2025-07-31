import os
import logging
import hashlib
import uuid
from typing import List, Dict, Any, Tuple
import asyncio
import aiofiles
from email import message_from_string
from email.policy import default as email_policy
import pdfplumber
from docx import Document as DocxDocument
from models.schemas import DocumentType

logger = logging.getLogger(__name__)

class DocumentIngestionService:
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.eml', '.txt'}
    
    async def process_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Process uploaded file and return chunks, metadatas, and IDs"""
        try:
            # Validate file size
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > int(os.getenv("MAX_FILE_SIZE_MB", "10")):
                raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({int(os.getenv('MAX_FILE_SIZE_MB', '10'))}MB)")
            
            # Determine file type
            file_extension = os.path.splitext(filename.lower())[1]
            if file_extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text_content = await self._extract_pdf_text(file_content)
                doc_type = DocumentType.PDF
            elif file_extension == '.docx':
                text_content = await self._extract_docx_text(file_content)
                doc_type = DocumentType.DOCX
            elif file_extension == '.eml':
                text_content = await self._extract_email_text(file_content)
                doc_type = DocumentType.EML
            elif file_extension == '.txt':
                text_content = file_content.decode('utf-8', errors='ignore')
                doc_type = DocumentType.TXT
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            if not text_content.strip():
                raise ValueError("No text content could be extracted from the file")
            
            # Create document ID
            document_id = self._generate_document_id(filename, file_content)
            
            # Split into chunks
            chunks = self._split_text_into_chunks(text_content)
            
            # Create metadata for each chunk
            metadatas = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "document_type": doc_type.value,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_size_mb": round(file_size_mb, 2),
                    "chunk_length": len(chunk)
                }
                
                metadatas.append(metadata)
                chunk_ids.append(chunk_id)
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created")
            return chunks, metadatas, chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {str(e)}")
            raise
    
    def _generate_document_id(self, filename: str, content: bytes) -> str:
        """Generate unique document ID based on filename and content hash"""
        content_hash = hashlib.md5(content).hexdigest()[:8]
        clean_filename = os.path.splitext(filename)[0][:20]  # Limit filename length
        return f"{clean_filename}_{content_hash}"
    
    async def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber"""
        temp_path = None
        try:
            # Save content to temporary file for pdfplumber
            temp_path = f"/tmp/{uuid.uuid4().hex}.pdf"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Processing PDF with {len(file_content)} bytes")
            
            # Extract text using pdfplumber
            text_content = []
            with pdfplumber.open(temp_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        logger.debug(f"Page {page_num + 1}: extracted {len(page_text) if page_text else 0} characters")
                        
                        if page_text and page_text.strip():
                            # Clean up the text
                            page_text = self._clean_text(page_text)
                            if page_text.strip():  # Only add if there's content after cleaning
                                text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            logger.debug(f"Page {page_num + 1}: found {len(tables)} tables")
                            for table_num, table in enumerate(tables):
                                if table and len(table) > 0:
                                    table_text = self._format_table(table)
                                    if table_text.strip():
                                        text_content.append(f"[Page {page_num + 1}, Table {table_num + 1}]\n{table_text}")
                                        
                    except Exception as page_error:
                        logger.warning(f"Error processing page {page_num + 1}: {str(page_error)}")
                        continue
            
            final_text = "\n\n".join(text_content)
            logger.info(f"PDF extraction completed: {len(final_text)} characters extracted")
            
            if not final_text.strip():
                raise ValueError("No text content could be extracted from the PDF. The PDF might be image-based or corrupted.")
            
            return final_text
            
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {str(e)}")
            raise ValueError(f"Could not extract text from PDF: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_path:
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    async def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX using python-docx"""
        try:
            # Save content to temporary file
            temp_path = f"/tmp/{uuid.uuid4().hex}.docx"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_content)
            
            # Extract text using python-docx
            doc = DocxDocument(temp_path)
            text_content = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(self._clean_text(para.text))
            
            # Extract tables
            for table_num, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    if any(row_data):  # Only add non-empty rows
                        table_data.append(row_data)
                
                if table_data:
                    table_text = self._format_table(table_data)
                    text_content.append(f"[Table {table_num + 1}]\n{table_text}")
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {str(e)}")
            raise ValueError(f"Could not extract text from DOCX: {str(e)}")
    
    async def _extract_email_text(self, file_content: bytes) -> str:
        """Extract text from email file"""
        try:
            # Parse email
            email_string = file_content.decode('utf-8', errors='ignore')
            msg = message_from_string(email_string, policy=email_policy)
            
            text_content = []
            
            # Extract headers
            headers = []
            for header in ['From', 'To', 'Cc', 'Subject', 'Date']:
                if msg.get(header):
                    headers.append(f"{header}: {msg.get(header)}")
            
            if headers:
                text_content.append("[Email Headers]\n" + "\n".join(headers))
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_content()
                        if body:
                            text_content.append("[Email Body]\n" + self._clean_text(body))
                    elif part.get_content_type() == "text/html":
                        # Basic HTML to text conversion
                        html_body = part.get_content()
                        if html_body:
                            import re
                            # Remove HTML tags
                            clean_html = re.sub(r'<[^>]+>', ' ', html_body)
                            clean_html = re.sub(r'\s+', ' ', clean_html).strip()
                            if clean_html:
                                text_content.append("[Email Body (HTML)]\n" + self._clean_text(clean_html))
            else:
                body = msg.get_content()
                if body:
                    text_content.append("[Email Body]\n" + self._clean_text(body))
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Failed to extract email text: {str(e)}")
            raise ValueError(f"Could not extract text from email: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text or text is None:
            return ""
            
        import re
        
        # Convert to string if not already
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove non-printable characters except common ones
        text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _format_table(self, table_data: List[List[str]]) -> str:
        """Format table data as text"""
        if not table_data:
            return ""
        
        # Simple table formatting
        formatted_rows = []
        for row in table_data:
            if not row:
                continue
            # Clean each cell and join with | separator
            clean_row = []
            for cell in row:
                if cell is not None:
                    clean_cell = str(cell).replace('\n', ' ').strip()
                    clean_row.append(clean_cell)
                else:
                    clean_row.append("")
            
            if any(clean_row):  # Only add rows that have some content
                formatted_rows.append(" | ".join(clean_row))
        
        return "\n".join(formatted_rows)
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []
        
        # Simple sentence-aware chunking
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_length + sentence_length > int(os.getenv("CHUNK_SIZE", "1000")) and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                words = current_chunk.split()
                
                # Add words from the end until we reach overlap size
                for word in reversed(words):
                    if overlap_length + len(word) + 1 <= int(os.getenv("CHUNK_OVERLAP", "200")):
                        overlap_sentences.insert(0, word)
                        overlap_length += len(word) + 1
                    else:
                        break
                
                current_chunk = " ".join(overlap_sentences) + " " + sentence if overlap_sentences else sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length + (1 if current_chunk else 0)
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return chunks

# Global ingestion service instance
ingestion_service = DocumentIngestionService()
