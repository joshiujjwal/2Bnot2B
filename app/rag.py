import logging
import os
import re
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize RAG system components"""
        logger.info("Initializing RAG system...")
        self.processed_documents = {}  # Track processed docs

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("documents")
        
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

             # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Create unique IDs and metadata
            ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": document_name,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "file_path": file_path
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to ChromaDB
            logger.info("Storing in vector database...")
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks from {document_name} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
       """Retrieve relevant documents using semantic search"""
       try:
           logger.info(f"Retrieving documents for query: {query[:50]}...")
           
           # Generate query embedding
           query_embedding = self.embedding_model.encode([query]).tolist()
           
           # Perform similarity search
           results = self.collection.query(
               query_embeddings=query_embedding,
               n_results=n_results,
               include=['documents', 'metadatas', 'distances']
           )
           
           # Format results
           retrieved_docs = []
           if results['documents'] and results['documents'][0]:
               for i, (doc, metadata, distance) in enumerate(zip(
                   results['documents'][0],
                   results['metadatas'][0],
                   results['distances'][0]
               )):
                   retrieved_docs.append({
                       'content': doc,
                       'metadata': metadata,
                       'distance': distance,
                       'similarity_score': 1 - distance,  # Convert distance to similarity
                       'rank': i + 1
                   })
           
           logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
           return retrieved_docs
           
       except Exception as e:
           logger.error(f"Error retrieving documents: {str(e)}")
           return []
   
    def query(self, question: str) -> Dict:
        """Query the RAG system with retrieval"""
        logger.info(f"Processing query: {question}")
        
        collection_count = self.collection.count()
        if collection_count == 0:
            return {
                'answer': 'No documents have been processed yet. Please upload some documents first.',
                'sources': [],
                'source_count': 0
            }
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve(question, n_results=5)
        
        if not relevant_docs:
            return {
                'answer': 'I could not find any relevant information in the uploaded documents to answer your question.',
                'sources': [],
                'source_count': 0
            }
        
        best_match = relevant_docs[0]
        similarity_score = best_match['similarity_score']
        
        if similarity_score > 0.7:
            confidence = "high"
        elif similarity_score > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        answer = f"""Based on the retrieved documents (confidence: {confidence}), here's the most relevant information I found:

    "{best_match['content'][:500]}..."

    This information comes from: {best_match['metadata']['source']} (chunk {best_match['metadata']['chunk_id'] + 1})

    Similarity score: {similarity_score:.3f}

    """

        return {
            'answer': answer,
            'sources': relevant_docs,
            'source_count': len(relevant_docs),
            'retrieval_info': {
                'total_chunks_searched': collection_count,
                'relevant_chunks_found': len(relevant_docs),
                'best_similarity_score': similarity_score
            }
        }