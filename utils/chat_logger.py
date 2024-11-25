from typing import List, Dict, Any
from datetime import datetime
import pymongo
from pymongo import MongoClient
import uuid

class ChatLogger:
    def __init__(self, mongodb_uri: str):
        """Initialize ChatLogger with MongoDB connection."""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.obi_chat_logs
        self.threads = self.db.threads
        
        # Create index on thread_id for faster queries
        self.threads.create_index("thread_id")
    
    def start_thread(self) -> str:
        """Start a new chat thread and return its ID."""
        thread_id = str(uuid.uuid4())
        self.threads.insert_one({
            "thread_id": thread_id,
            "messages": []
        })
        return thread_id
    
    def log_message(self, thread_id: str, role: str, content: str) -> None:
        """Log a message to a specific thread."""
        message = {
            "timestamp": datetime.utcnow(),
            "role": role,
            "content": content
        }
        
        self.threads.update_one(
            {"thread_id": thread_id},
            {"$push": {"messages": message}}
        )
    
    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Retrieve a complete thread by its ID."""
        return self.threads.find_one({"thread_id": thread_id})
    
    def get_all_threads(self) -> List[Dict[str, Any]]:
        """Retrieve all chat threads."""
        return list(self.threads.find({}))
    
    def get_threads_in_timerange(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Retrieve threads with messages in the specified time range."""
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

# Example retrieval script:
"""
from datetime import datetime, timedelta
from chat_logger import ChatLogger

def analyze_chat_threads():
    # Initialize ChatLogger with your MongoDB URI
    logger = ChatLogger("your_mongodb_uri")
    
    # Get all threads
    all_threads = logger.get_all_threads()
    print(f"Total threads: {len(all_threads)}")
    
    # Get threads from last 24 hours
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)
    recent_threads = logger.get_threads_in_timerange(start_date, end_date)
    print(f"Threads in last 24 hours: {len(recent_threads)}")
    
    # Print details of each thread
    for thread in recent_threads:
        print(f"\nThread ID: {thread['thread_id']}")
        for msg in thread['messages']:
            print(f"{msg['timestamp']} - {msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    analyze_chat_threads()
"""
