import logging
import os
import re
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        """Initialize RAG system components"""
        logger.info("Initializing RAG system...")
        self.processed_documents = {}  # Track processed docs
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_boundary = max(last_period, last_newline)
                
                if last_boundary > chunk_size * 0.5:  # Ensure chunk isn't too small
                    chunk = chunk[:last_boundary + 1]
                    end = start + last_boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
        
    def add_document(self, file_path: str, document_name: str = None) -> bool:
        """Add a document to the knowledge base"""
        try:
            if not document_name:
                document_name = os.path.basename(file_path)
                
            logger.info(f"Processing document: {document_name}")
            
            # Extract text
            raw_text = self.extract_text_from_file(file_path)
            if not raw_text.strip():
                logger.warning(f"No text extracted from {document_name}")
                return False
            
            # Preprocess text
            clean_text = self.preprocess_text(raw_text)
            
            # Create chunks
            chunks = self.chunk_text(clean_text)
            
            logger.info(f"Successfully processed {document_name}: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False
    