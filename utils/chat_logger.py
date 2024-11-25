from typing import List, Dict, Any
from datetime import datetime
import uuid
from pymongo import MongoClient
import logging
import certifi
import ssl

logger = logging.getLogger(__name__)

class ChatLogger:
    def __init__(self, mongodb_uri: str):
        """Initialize ChatLogger with MongoDB connection."""
        try:
            # Add TLS/SSL settings to connection
            if not mongodb_uri:
                raise ValueError("MongoDB URI is required")
            
            # Configure SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Use certifi for SSL certificate verification
            self.client = MongoClient(
                mongodb_uri,
                ssl=True,
                ssl_cert_reqs=ssl.CERT_NONE,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                retryWrites=True,
                w="majority"
            )
            self.db = self.client.obi_chat_logs
            self.threads = self.db.threads
            
            # Create index on thread_id for faster queries
            self.threads.create_index("thread_id")
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {str(e)}")
            # Don't raise the error, just log it and continue without MongoDB
            self.client = None
            self.db = None
            self.threads = None
    
    def start_thread(self) -> str:
        """Start a new chat thread and return its ID."""
        thread_id = str(uuid.uuid4())
        if not self.threads:
            logger.warning("MongoDB not available, skipping thread creation")
            return thread_id
            
        try:
            self.threads.insert_one({
                "thread_id": thread_id,
                "messages": []
            })
            logger.info(f"Started new thread: {thread_id}")
        except Exception as e:
            logger.error(f"Failed to start thread: {str(e)}")
        
        return thread_id
    
    def log_message(self, thread_id: str, role: str, content: str) -> None:
        """Log a message to a specific thread."""
        if not self.threads:
            logger.warning("MongoDB not available, skipping message logging")
            return
            
        message = {
            "timestamp": datetime.utcnow(),
            "role": role,
            "content": content
        }
        
        try:
            self.threads.update_one(
                {"thread_id": thread_id},
                {"$push": {"messages": message}}
            )
            logger.debug(f"Logged message to thread {thread_id}")
        except Exception as e:
            logger.error(f"Failed to log message: {str(e)}")
    
    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Retrieve a complete thread by its ID."""
        if not self.threads:
            logger.warning("MongoDB not available, cannot retrieve thread")
            return None
            
        try:
            return self.threads.find_one({"thread_id": thread_id})
        except Exception as e:
            logger.error(f"Failed to retrieve thread {thread_id}: {str(e)}")
            return None
    
    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Retrieve all chat threads."""
        if not self.threads:
            logger.warning("MongoDB not available, cannot retrieve threads")
            return []
            
        try:
            return list(self.threads.find({}))
        except Exception as e:
            logger.error(f"Failed to retrieve threads: {str(e)}")
            return []
    
    def get_threads_in_timerange(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Retrieve threads with messages in the specified time range."""
        if not self.threads:
            logger.warning("MongoDB not available, cannot retrieve threads")
            return []
            
        try:
            return list(self.threads.find({
                "messages": {
                    "$elemMatch": {
                        "timestamp": {
                            "$gte": start_date,
                            "$lte": end_date
                        }
                    }
                }
            }))
        except Exception as e:
            logger.error(f"Failed to retrieve threads in timerange: {str(e)}")
            return []
    
    def __del__(self):
        """Ensure MongoDB connection is closed."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
                logger.info("Closed MongoDB connection")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {str(e)}")
