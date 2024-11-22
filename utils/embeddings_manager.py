# Patch SQLite before chromadb import
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Now we can safely import everything else
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, TypedDict, Set, cast
import logging
from tqdm import tqdm
import os
import json
import numpy as np
import torch
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO level logging is enabled

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]

class ChromaMetadata(TypedDict):
    source: str
    chunk_id: int
    file_path: str

class EmbeddingsManager:
    def __init__(self, model_name: str, db_path: str):
        """Initialize the embeddings manager with a specified model and database path."""
        logger.info(f"Initializing EmbeddingsManager with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.processed_files_path = os.path.join(db_path, "processed_files.json")
        
        # Ensure the database directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB with persistent settings
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        # Use persistent client with settings
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=settings
        )
        
        # Get or create collection using a single operation
        collection_name = "drivers_license_docs"
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Drivers license documents collection"}
            )
            logger.info(f"Successfully initialized collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
        
        # Load processed files and perform cleanup
        self.processed_files = self._load_processed_files()
        logger.info(f"Loaded processed files: {self.processed_files}")
        self._cleanup_missing_files()
        
        # Process any new files in the documents directory
        self.process_new_files()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text from PDF conversion artifacts."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Add spaces between numbers and letters
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
        
        # Fix common PDF conversion artifacts
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase
        text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)  # ABCdef
        
        # Remove space between dollar sign and number
        text = re.sub(r'\$\s+(\d)', r'$\1', text)
        
        # Clean up any multiple spaces created
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _load_processed_files(self) -> Set[str]:
        """Load the set of already processed files."""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading processed files list: {e}")
                return set()
        return set()

    def _save_processed_files(self) -> None:
        """Save the set of processed files."""
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.processed_files_path, 'w') as f:
            json.dump(list(self.processed_files), f)
        logger.info(f"Saved processed files: {self.processed_files}")

    def _cleanup_missing_files(self) -> None:
        """Remove entries for files that no longer exist in the documents directory."""
        if not self.processed_files:
            return

        # Get the actual files in the documents directory
        docs_dir = os.getenv('DOCUMENTS_PATH', './data/drivers_license_docs')
        os.makedirs(docs_dir, exist_ok=True)  # Ensure documents directory exists
        existing_files = set(f for f in os.listdir(docs_dir) if f.endswith('.txt'))
        logger.info(f"Found existing files: {existing_files}")
        
        # Find files that have been processed but no longer exist
        missing_files = self.processed_files - existing_files
        
        if missing_files:
            logger.info(f"Found {len(missing_files)} files to clean up: {missing_files}")
            
            for filename in missing_files:
                try:
                    # Delete from the Chroma DB
                    results = self.collection.get(
                        where={"source": filename}
                    )
                    
                    if results and results.get('ids'):
                        self.collection.delete(
                            ids=results['ids']
                        )
                        logger.info(f"Deleted entries for {filename} from Chroma DB")
                    
                    # Remove from processed files list
                    self.processed_files.remove(filename)
                    logger.info(f"Removed {filename} from processed files list")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up {filename}: {e}")
            
            # Save updated processed files list
            self._save_processed_files()
            logger.info("Cleanup completed")

    def process_text_file(self, file_path: str, chunk_size: int = 200) -> List[Document]:
        """Process a text file into chunks and create Document objects."""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text before processing
            text = self._clean_text(text)
            
            # Split text into sentences first
            sentences = text.split('.')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:  # Skip empty sentences
                    continue
                    
                # Add period back to sentence
                sentence = sentence + '.'
                
                sentence_length = len(sentence)
                if sentence_length > chunk_size:
                    # If a single sentence is longer than chunk_size, make it its own chunk
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    chunks.append(sentence)
                elif current_length + sentence_length > chunk_size:
                    # If adding this sentence would exceed chunk_size, start a new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add any remaining chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Create Document objects for each chunk
            filename = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                # Clean each chunk again to ensure consistency
                chunk = self._clean_text(chunk)
                doc = Document(
                    text=chunk,
                    metadata={
                        'source': filename,
                        'chunk_id': i,
                        'file_path': file_path
                    }
                )
                documents.append(doc)
            
            logger.info(f"Processed {filename} into {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return documents

    def process_new_files(self) -> None:
        """Check for and process any new text files in the documents directory."""
        docs_dir = os.getenv('DOCUMENTS_PATH', './data/drivers_license_docs')
        logger.info(f"Checking for new files in {docs_dir}")
        
        # Get list of text files
        text_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
        logger.info(f"Found text files: {text_files}")
        
        # Find new files
        new_files = [f for f in text_files if f not in self.processed_files]
        
        if new_files:
            logger.info(f"Found {len(new_files)} new files to process: {new_files}")
            all_documents = []
            
            for filename in new_files:
                file_path = os.path.join(docs_dir, filename)
                documents = self.process_text_file(file_path)
                all_documents.extend(documents)
            
            if all_documents:
                self.add_documents(all_documents)
                logger.info(f"Successfully processed {len(new_files)} new files")
        else:
            logger.info("No new files to process")

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """Generate embeddings for new documents and store them in the Chroma database."""
        if not documents:
            return

        # Group documents by source file
        documents_by_file: Dict[str, List[Document]] = {}
        for doc in documents:
            source = str(doc.metadata['source'])
            if source not in documents_by_file:
                documents_by_file[source] = []
            documents_by_file[source].append(doc)

        # Process only new files
        new_documents = []
        for source, docs in documents_by_file.items():
            if source not in self.processed_files:
                new_documents.extend(docs)
                logger.info(f"Found new file to process: {source}")

        if not new_documents:
            logger.info("No new documents to process")
            return

        logger.info(f"Processing {len(new_documents)} documents from {len(documents_by_file)} new files")
        
        # Process in batches
        for i in tqdm(range(0, len(new_documents), batch_size), desc="Creating embeddings"):
            batch = new_documents[i:i + batch_size]
            
            texts = [doc.text for doc in batch]
            metadatas: List[ChromaMetadata] = []
            
            for doc in batch:
                metadata: ChromaMetadata = {
                    'source': str(doc.metadata['source']),
                    'chunk_id': int(doc.metadata['chunk_id']),
                    'file_path': str(doc.metadata['file_path'])
                }
                metadatas.append(metadata)
                
            ids = [f"{metadata['source']}_{metadata['chunk_id']}" for metadata in metadatas]
            
            try:
                # Generate embeddings using sentence transformer
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
                # Convert numpy array to list of lists
                embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else [
                    tensor.detach().cpu().numpy().tolist() for tensor in embeddings
                ]
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings_list,
                    documents=texts,
                    metadatas=metadatas,  # type: ignore
                    ids=ids
                )
                logger.debug(f"Successfully added batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error adding batch to collection: {str(e)}")
                raise

        # Update processed files list
        for source in documents_by_file.keys():
            if source not in self.processed_files:
                self.processed_files.add(source)
                logger.info(f"Marked {source} as processed")

        # Save updated processed files list
        self._save_processed_files()
    
    def get_collection(self) -> Any:
        """Return the Chroma collection for querying."""
        return self.collection
