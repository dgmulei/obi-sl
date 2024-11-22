from typing import List, Dict, Any, Union, Optional, Sequence, cast
from dataclasses import dataclass
import sys
from typing_extensions import TypedDict, NotRequired
import logging

logger = logging.getLogger(__name__)

class DocumentMetadata(TypedDict):
    source: NotRequired[str]
    page: NotRequired[int]
    
@dataclass
class QueryResult:
    text: str
    metadata: Dict[str, Any]
    distance: float

class QueryEngine:
    def __init__(self, collection):
        """Initialize QueryEngine with an existing collection."""
        self.collection = collection
        if self.collection is None:
            raise ValueError("Collection cannot be None")
        logger.info("Successfully initialized QueryEngine with existing collection")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        if not texts or not metadatas or not ids:
            return
            
        # Add documents to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 10) -> List[QueryResult]:
        try:
            # Handle empty collection case
            count = self.collection.count()
            if count == 0:
                logger.info("Collection is empty, returning no results")
                return []
                
            # Log the query for debugging
            logger.info(f"Querying with text: {query_text}")
                
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count),  # Don't request more results than documents
                include=["documents", "metadatas", "distances"]
            )
            
            query_results: List[QueryResult] = []
            
            # Check if we have valid results
            if not results or not isinstance(results, dict):
                logger.warning("Query returned no results")
                return []
            
            # Get the results with safe defaults
            result_dict = cast(Dict[str, List[List[Any]]], results)
            documents_list = result_dict.get("documents", [[]])
            metadatas_list = result_dict.get("metadatas", [[]])
            distances_list = result_dict.get("distances", [[]])
            
            # Log the raw results for debugging
            logger.info("Raw query results: %s", result_dict)
            
            # Ensure we have at least one result
            if not documents_list or not documents_list[0]:
                return []
                
            # Get the first (and only) set of results
            documents = documents_list[0]
            metadatas = metadatas_list[0] if metadatas_list else [{}] * len(documents)
            distances = distances_list[0] if distances_list else [0.0] * len(documents)
            
            # Create QueryResult objects
            for doc, meta, dist in zip(documents, metadatas, distances):
                query_results.append(
                    QueryResult(
                        text=str(doc),
                        metadata=meta or {},
                        distance=float(dist)
                    )
                )
            
            # Log the processed results for debugging
            for result in query_results:
                logger.info("Query result - Text: %s..., Metadata: %s, Distance: %f", 
                          result.text[:100] if result.text else "", 
                          result.metadata, 
                          result.distance)
            
            return query_results
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return []
