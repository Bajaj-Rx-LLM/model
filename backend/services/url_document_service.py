import logging
import httpx
import asyncio
from typing import Optional, Tuple
from urllib.parse import urlparse
import os
import tempfile

logger = logging.getLogger(__name__)

class URLDocumentService:
    """Service for downloading and processing documents from URLs"""
    
    def __init__(self):
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024  # Convert to bytes
        self.timeout = 60  # seconds
    
    async def download_document(self, url: str) -> Tuple[bytes, str]:
        """
        Download document from URL with Azure Blob Storage support
        
        Args:
            url: URL to download from
            
        Returns:
            Tuple of (file_content, filename)
        """
        try:
            logger.info(f"Downloading document from: {url}")
            
            # Parse URL to extract filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # If no filename in URL, use a default
            if not filename or '.' not in filename:
                filename = "document.pdf"
            
            # Enhanced headers for Azure Blob Storage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Try multiple download strategies for Azure Blob Storage
            strategies = [
                self._download_with_azure_headers,
                self._download_with_basic_headers,
                self._download_without_headers
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    logger.info(f"Trying download strategy {i+1}/{len(strategies)}")
                    file_content = await strategy(url, headers)
                    
                    # Validate file content
                    if len(file_content) == 0:
                        raise ValueError("Downloaded file is empty")
                    
                    # Check file size
                    if len(file_content) > self.max_file_size:
                        raise ValueError(f"Downloaded file size ({len(file_content) / 1024 / 1024:.1f}MB) exceeds maximum allowed size")
                    
                    logger.info(f"Successfully downloaded {len(file_content)} bytes as '{filename}'")
                    return file_content, filename
                    
                except Exception as e:
                    logger.warning(f"Download strategy {i+1} failed: {str(e)}")
                    if i == len(strategies) - 1:  # Last strategy failed
                        raise ValueError(f"All download strategies failed. Last error: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise ValueError(f"Failed to download document: {str(e)}")
    
    async def _download_with_azure_headers(self, url: str, base_headers: dict) -> bytes:
        """Download with Azure-specific headers"""
        headers = base_headers.copy()
        headers.update({
            'x-ms-version': '2020-04-08',
            'x-ms-blob-type': 'BlockBlob',
            'Referer': 'https://hackrx.blob.core.windows.net/'
        })
        
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers=headers
        ) as client:
            response = await client.get(url)
            
            if response.status_code == 403:
                raise ValueError("Access denied (HTTP 403). The document URL may have expired or requires authentication.")
            elif response.status_code == 404:
                raise ValueError("Document not found (HTTP 404)")
            elif response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
            
            return response.content
    
    async def _download_with_basic_headers(self, url: str, base_headers: dict) -> bytes:
        """Download with basic browser headers"""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers=base_headers
        ) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            
            return response.content
    
    async def _download_without_headers(self, url: str, base_headers: dict) -> bytes:
        """Download without custom headers"""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True
        ) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            
            return response.content
    
    async def process_url_document(self, url: str, ingestion_service) -> Tuple[str, int]:
        """
        Download and process document from URL
        
        Args:
            url: URL to download from
            ingestion_service: Service to process the document
            
        Returns:
            Tuple of (document_id, chunks_count)
        """
        try:
            # Download the document
            file_content, filename = await self.download_document(url)
            
            # Process the document using ingestion service
            chunks, metadatas, chunk_ids = await ingestion_service.process_uploaded_file(
                file_content=file_content,
                filename=filename
            )
            
            if not chunks:
                raise ValueError("No content could be extracted from the downloaded document")
            
            # Add to vector database
            from services.chromadb_service import chroma_service
            success = await chroma_service.add_documents(chunks, metadatas, chunk_ids)
            
            if not success:
                raise ValueError("Failed to add document to vector database")
            
            # Extract document ID from metadata
            document_id = metadatas[0]["document_id"] if metadatas else "unknown"
            
            logger.info(f"Successfully processed URL document: {document_id} with {len(chunks)} chunks")
            return document_id, len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process URL document: {str(e)}")
            raise

# Global URL document service instance
url_document_service = URLDocumentService()
