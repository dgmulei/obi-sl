from google.cloud import storage
from google.api_core import retry
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import os
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatStorage:
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
            logger.info(f"Successfully initialized ChatStorage with bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatStorage: {str(e)}")
            raise
    
    @retry.Retry(predicate=retry.if_transient_error)
    def save_thread(self, messages: List[Dict[str, Any]], thread_id: str = None) -> str:
        """
        Save a chat thread to GCS with retry logic for transient errors.
        
        Args:
            messages: List of message dictionaries with role, content, and timestamp
            thread_id: Optional thread ID. If not provided, a new UUID will be generated
            
        Returns:
            thread_id: The ID of the saved thread
            
        Raises:
            ValueError: If messages is empty or invalid
            Exception: For other storage-related errors
        """
        try:
            if not messages:
                raise ValueError("Cannot save empty message list")
                
            if not thread_id:
                thread_id = str(uuid.uuid4())
                
            thread_data = {
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "messages": messages
            }
            
            # Convert thread data to JSON
            json_data = json.dumps(thread_data, ensure_ascii=False)
            
            # Save to GCS
            blob = self.bucket.blob(f"chat-histories/{thread_id}.json")
            blob.upload_from_string(
                json_data,
                content_type='application/json'
            )
            
            logger.info(f"Successfully saved thread {thread_id}")
            return thread_id
            
        except ValueError as e:
            logger.error(f"Invalid input for thread save: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to save thread {thread_id}: {str(e)}")
            raise
    
    @retry.Retry(predicate=retry.if_transient_error)
    def update_thread(self, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Update an existing chat thread with new messages.
        
        Args:
            thread_id: The ID of the thread to update
            messages: Complete list of messages (including new ones)
            
        Raises:
            ValueError: If thread_id is invalid or messages is empty
            Exception: For storage-related errors
        """
        if not thread_id:
            raise ValueError("thread_id is required for updates")
            
        try:
            # Save thread with existing ID
            self.save_thread(messages, thread_id)
            logger.info(f"Successfully updated thread {thread_id}")
        except Exception as e:
            logger.error(f"Failed to update thread {thread_id}: {str(e)}")
            raise
    
    def format_message(self, role: str, content: str) -> Dict[str, Any]:
        """
        Format a message for storage.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            Formatted message dictionary
            
        Raises:
            ValueError: If role or content is invalid
        """
        if not role or not content:
            raise ValueError("Both role and content are required")
            
        if role not in ['user', 'assistant']:
            raise ValueError("Invalid role. Must be 'user' or 'assistant'")
            
        return {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
