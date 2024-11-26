from google.cloud import storage
from google.api_core import retry
from google.cloud.exceptions import NotFound
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timezone
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRetrieval:
    def __init__(self):
        """Initialize GCS client and bucket."""
        try:
            self.client = storage.Client()
            bucket_name = os.getenv('GCS_BUCKET_NAME')
            if not bucket_name:
                raise ValueError("GCS_BUCKET_NAME environment variable not set")
            self.bucket = self.client.bucket(bucket_name)
            # Verify bucket exists and we have access
            if not self.bucket.exists():
                raise ValueError(f"Bucket {bucket_name} does not exist or is not accessible")
            logger.info(f"Successfully initialized ChatRetrieval with bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatRetrieval: {str(e)}")
            raise

    @retry.Retry(predicate=retry.if_transient_error)
    def get_chat_thread(self, thread_id: str) -> Dict[str, Any]:
        """
        Retrieve a single chat thread by ID with retry logic for transient errors.
        
        Args:
            thread_id: The unique identifier for the chat thread
            
        Returns:
            Dict containing the thread data
            
        Raises:
            ValueError: If thread_id is invalid
            NotFound: If the thread doesn't exist
            Exception: For other storage-related errors
        """
        if not thread_id:
            raise ValueError("thread_id is required")
            
        try:
            blob = self.bucket.blob(f"chat-histories/{thread_id}.json")
            if not blob.exists():
                raise NotFound(f"Thread {thread_id} not found")
                
            content = blob.download_as_string()
            thread_data = json.loads(content)
            logger.info(f"Successfully retrieved thread {thread_id}")
            return thread_data
            
        except NotFound as e:
            logger.error(f"Thread not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve thread {thread_id}: {str(e)}")
            raise

    @retry.Retry(predicate=retry.if_transient_error)
    def get_threads_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve all chat threads within a date range with retry logic.
        
        Args:
            start_date: Start of date range (timezone-aware)
            end_date: End of date range (timezone-aware)
            
        Returns:
            List of thread data dictionaries
            
        Raises:
            ValueError: If date range is invalid
            Exception: For storage-related errors
        """
        if not start_date or not end_date:
            raise ValueError("Both start_date and end_date are required")
            
        if end_date < start_date:
            raise ValueError("end_date must be after start_date")
            
        try:
            threads = []
            # List all blobs in chat-histories/
            blobs = self.bucket.list_blobs(prefix="chat-histories/")
            
            for blob in blobs:
                try:
                    content = blob.download_as_string()
                    thread = json.loads(content)
                    
                    # Parse thread timestamp
                    thread_time = datetime.fromisoformat(thread['timestamp'].replace('Z', '+00:00'))
                    
                    # Check if thread is within date range
                    if start_date <= thread_time <= end_date:
                        threads.append(thread)
                except Exception as e:
                    # Log error but continue processing other threads
                    logger.error(f"Error processing thread from blob {blob.name}: {str(e)}")
                    continue
            
            logger.info(f"Retrieved {len(threads)} threads between {start_date} and {end_date}")
            return threads
            
        except Exception as e:
            logger.error(f"Failed to retrieve threads by date range: {str(e)}")
            raise

    def get_recent_threads(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent chat threads.
        
        Args:
            limit: Maximum number of threads to retrieve
            
        Returns:
            List of thread data dictionaries, sorted by timestamp (newest first)
            
        Raises:
            ValueError: If limit is invalid
            Exception: For storage-related errors
        """
        if limit < 1:
            raise ValueError("limit must be positive")
            
        try:
            threads = []
            blobs = self.bucket.list_blobs(prefix="chat-histories/")
            
            for blob in blobs:
                try:
                    content = blob.download_as_string()
                    thread = json.loads(content)
                    threads.append(thread)
                except Exception as e:
                    logger.error(f"Error processing thread from blob {blob.name}: {str(e)}")
                    continue
            
            # Sort threads by timestamp (newest first) and limit results
            threads.sort(key=lambda x: x['timestamp'], reverse=True)
            threads = threads[:limit]
            
            logger.info(f"Retrieved {len(threads)} recent threads")
            return threads
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent threads: {str(e)}")
            raise
