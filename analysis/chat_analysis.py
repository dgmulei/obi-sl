from datetime import datetime, timezone, timedelta
import sys
import os
from typing import List, Dict, Any
from collections import Counter
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_retrieval import ChatRetrieval

def analyze_chat_threads(days: int = 7) -> Dict[str, Any]:
    """
    Analyze chat threads from the last N days.
    
    Args:
        days: Number of days to analyze (default: 7)
    
    Returns:
        Dictionary containing analysis results
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    print(f"\nAnalyzing chat threads from {start.date()} to {end.date()}")
    
    # Initialize chat retrieval
    try:
        retriever = ChatRetrieval()
        # Retrieve threads
        threads = retriever.get_threads_by_date_range(start, end)
        print(f"Found {len(threads)} threads")
    except Exception as e:
        print(f"\n❌ Error retrieving threads: {str(e)}")
        return None
    
    # Sort threads by timestamp
    threads.sort(key=lambda x: x['timestamp'])
    
    # First display complete thread contents
    print("\n=== Complete Thread Contents (Chronological Order) ===")
    for thread in threads:
        print(f"\nThread ID: {thread['thread_id']}")
        print(f"Timestamp: {thread['timestamp']}")
        print("Messages:")
        for msg in thread['messages']:
            print(f"[{msg.get('role', 'unknown')}]: {msg.get('content', 'no content')}")
        print("-" * 80)
    
    # Initialize analysis results
    results = {
        "total_threads": len(threads),
        "total_messages": 0,
        "avg_messages_per_thread": 0,
        "avg_thread_duration": timedelta(0),
        "common_topics": [],
        "thread_durations": [],
        "hourly_activity": Counter(),
        "threads_by_date": Counter()
    }
    
    # Topic keywords to track
    topics = {
        "renewal": ["renew", "renewal", "expire", "expiration"],
        "documents": ["document", "documentation", "paperwork", "forms"],
        "fees": ["fee", "cost", "price", "payment"],
        "scheduling": ["schedule", "appointment", "book", "time"],
        "requirements": ["require", "requirement", "need", "necessary"],
        "technical": ["website", "online", "portal", "error"]
    }
    
    topic_counts = Counter()
    
    # Analyze each thread
    for thread in threads:
        messages = thread["messages"]
        results["total_messages"] += len(messages)
        
        # Thread dates
        thread_time = datetime.fromisoformat(thread["timestamp"].rstrip('Z'))
        thread_date = thread_time.date()
        results["threads_by_date"][thread_date] += 1
        
        # Message timing
        if messages:
            try:
                # Handle timestamps with or without Z suffix
                first_msg_time = datetime.fromisoformat(messages[0].get("timestamp", "").rstrip('Z'))
                last_msg_time = datetime.fromisoformat(messages[-1].get("timestamp", "").rstrip('Z'))
                
                # Track hourly activity
                results["hourly_activity"][first_msg_time.hour] += 1
                
                # Calculate thread duration
                duration = last_msg_time - first_msg_time
                results["thread_durations"].append(duration)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process timestamps for thread {thread.get('thread_id')}: {e}")
                continue
            
            # Topic analysis
            thread_text = " ".join(msg["content"].lower() for msg in messages)
            for topic, keywords in topics.items():
                if any(keyword in thread_text for keyword in keywords):
                    topic_counts[topic] += 1
    
    # Calculate averages
    if results["total_threads"] > 0:
        results["avg_messages_per_thread"] = results["total_messages"] / results["total_threads"]
        if results["thread_durations"]:
            total_duration = sum(results["thread_durations"], timedelta())
            results["avg_thread_duration"] = total_duration / len(results["thread_durations"])
    
    # Get top topics
    results["common_topics"] = topic_counts.most_common()
    
    # Format results for display
    print("\n=== Analysis Results ===")
    print(f"Total Threads: {results['total_threads']}")
    print(f"Total Messages: {results['total_messages']}")
    print(f"Average Messages per Thread: {results['avg_messages_per_thread']:.1f}")
    print(f"Average Thread Duration: {results['avg_thread_duration']}")
    
    if results["common_topics"]:
        print("\nMost Common Topics:")
        for topic, count in results["common_topics"]:
            percentage = (count / results["total_threads"]) * 100
            print(f"- {topic}: {count} threads ({percentage:.1f}%)")
    
    if results["hourly_activity"]:
        print("\nHourly Activity (UTC):")
        max_count = max(results["hourly_activity"].values()) if results["hourly_activity"] else 1
        for hour in sorted(results["hourly_activity"]):
            count = results["hourly_activity"][hour]
            bar = "█" * (count * 50 // max_count)
            print(f"{hour:02d}:00 {bar} ({count})")
    
    if results["threads_by_date"]:
        print("\nDaily Thread Count:")
        for date in sorted(results["threads_by_date"]):
            count = results["threads_by_date"][date]
            print(f"{date}: {count} threads")
    
    return results

if __name__ == "__main__":
    # Check for GCS credentials
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
        print("\n❌ Error: GCS credentials not found in environment variables")
        print("Please set either GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_JSON")
        sys.exit(1)
        
    if not os.getenv('GCS_BUCKET_NAME'):
        print("\n❌ Error: GCS_BUCKET_NAME not found in environment variables")
        sys.exit(1)
    
    # Get analysis timeframe from command line argument
    days = 7
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"\n❌ Error: Invalid number of days: {sys.argv[1]}")
            sys.exit(1)
    
    analyze_chat_threads(days)
