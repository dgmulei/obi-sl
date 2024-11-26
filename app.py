# Patch SQLite before any other imports
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Now we can safely import everything else
import streamlit as st
import os
import yaml
import logging
import json
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict, List, Optional, cast
from utils.conversation_manager import ConversationManager, SessionManager, ConversationContext
from utils.embeddings_manager import EmbeddingsManager
from utils.query_engine import QueryEngine
from utils.chat_storage import ChatStorage
from datetime import datetime
import tempfile

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ===== TYPE DEFINITIONS =====
class UserProfile(TypedDict):
    personal: Dict[str, Any]
    addresses: Dict[str, Any]
    license: Dict[str, Any]
    documentation: Dict[str, Any]
    additional: Dict[str, Any]
    health: Dict[str, Any]
    payment: Dict[str, Any]
    metadata: Dict[str, Any]

class UserProfiles(TypedDict):
    users: List[UserProfile]

# ===== INITIALIZATION FUNCTIONS =====
def setup_gcp_credentials() -> None:
    """Set up Google Cloud credentials from Streamlit secrets or local environment."""
    try:
        # First try to get credentials from environment variable
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.info("Using GCP credentials from environment variable")
            return

        # Then try to use Streamlit secrets if available
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            logger.info("Setting up GCP credentials from Streamlit secrets")
            # Create a temporary file to store the credentials
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(dict(st.secrets['gcp_service_account']), f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
            logger.info("Successfully set up GCP credentials from Streamlit secrets")
            return

        # If neither is available, raise an error
        raise ValueError("No GCP credentials found in environment or Streamlit secrets")
    except Exception as e:
        logger.error(f"Failed to set up GCP credentials: {str(e)}")
        raise

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        os.getenv('CHROMA_DB_PATH', './chroma_db'),
        os.getenv('DOCUMENTS_PATH', './data/drivers_license_docs'),
        '.streamlit'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def load_user_profiles() -> UserProfiles:
    """Load user profiles from YAML file."""
    empty_profiles: UserProfiles = {"users": []}
    try:
        with open('user-profiles-yaml.txt', 'r') as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict) or 'users' not in data:
                return empty_profiles
            return cast(UserProfiles, data)
    except Exception as e:
        logger.error(f"Error loading user profiles: {e}")
        return empty_profiles

def get_processed_files_path() -> str:
    """Get the path to the processed files list."""
    db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    return os.path.join(db_path, "processed_files.json")

def check_for_new_files() -> bool:
    """Check if there are any new files that need to be processed."""
    docs_dir = os.getenv('DOCUMENTS_PATH', './data/drivers_license_docs')
    processed_files_path = get_processed_files_path()
    
    # Get current text files
    text_files = set(f for f in os.listdir(docs_dir) if f.endswith('.txt'))
    
    # Get previously processed files
    processed_files = set()
    if os.path.exists(processed_files_path):
        try:
            import json
            with open(processed_files_path, 'r') as f:
                processed_files = set(json.load(f))
        except Exception as e:
            logger.warning(f"Error loading processed files list: {e}")
    
    # Check if there are any new files
    return bool(text_files - processed_files)

# ===== CACHED RESOURCES =====
@st.cache_resource(show_spinner=False)
def get_embeddings_manager(_has_new_files: bool) -> EmbeddingsManager:
    """Get or create embeddings manager. The _has_new_files parameter forces cache invalidation when new files are detected."""
    model_name = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    return EmbeddingsManager(model_name=model_name, db_path=db_path)

@st.cache_resource
def get_chat_storage() -> Optional[ChatStorage]:
    """Initialize chat storage with proper error handling."""
    try:
        # Ensure GCP credentials are set up
        setup_gcp_credentials()
        
        # Get bucket name from environment or Streamlit secrets
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if not bucket_name and hasattr(st, 'secrets'):
            bucket_name = st.secrets.get('GCS_BUCKET_NAME')
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME not found in environment or secrets")
        
        # Set environment variable for ChatStorage
        os.environ['GCS_BUCKET_NAME'] = bucket_name
        
        # Initialize storage
        storage = ChatStorage()
        logger.info("Successfully initialized chat storage")
        return storage
    except Exception as e:
        logger.error(f"Failed to initialize chat storage: {str(e)}")
        # Return None instead of raising to allow app to continue without storage
        return None

@st.cache_resource
def initialize_components():
    logger.info("Starting initialization...")
    ensure_directories()
    
    # Check for new files and initialize embeddings manager
    has_new_files = check_for_new_files()
    logger.info(f"New files detected: {has_new_files}")
    embeddings_manager = get_embeddings_manager(has_new_files)
    
    # Get the collection from embeddings manager
    collection = embeddings_manager.get_collection()
    if collection is None:
        raise ValueError("Failed to initialize ChromaDB collection")
    
    # Initialize query engine with the existing collection
    query_engine = QueryEngine(collection=collection)
    
    # Check for Anthropic API key
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key and hasattr(st, 'secrets'):
        anthropic_api_key = st.secrets.get('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        raise ValueError("Anthropic API key not found")
    
    # Initialize chat storage (may be None if initialization fails)
    chat_storage = get_chat_storage()
    
    # Initialize conversation manager
    conversation_manager = ConversationManager(
        query_engine=query_engine,
        api_key=anthropic_api_key,
        chat_storage=chat_storage  # May be None, which is handled by ConversationManager
    )
    
    return embeddings_manager, query_engine, conversation_manager

# ===== HELPER FUNCTIONS =====
def display_chat_messages(messages):
    """Display chat messages excluding system messages."""
    for message in messages:
        if message.role != "system" and message.visible:
            with st.chat_message(message.role):
                st.markdown(message.content)

def process_user_message(message: str, conversation_manager: ConversationManager, context, visible: bool = True):
    """Process user message and get response."""
    response = conversation_manager.get_response(message, context, visible=visible)
    return True

def display_user_info(user: Dict[str, Any]):
    """Display user information in a collapsible expander."""
    with st.expander("User Information", expanded=False):
        st.write(f"Name: {user['personal']['full_name']}")
        st.write(f"DOB: {user['personal']['dob']}")
        st.write(f"License Number: {user['license']['current']['number']}")
        st.write(f"Expiration: {user['license']['current']['expiration']}")

# ===== MAIN APPLICATION =====
def main():
    st.set_page_config(
        page_title="Obi - Driver's License Renewal Guide",
        page_icon="🪪",
        layout="wide"
    )
    
    # Initialize session states for both citizens
    if 'citizen1_context' not in st.session_state:
        st.session_state.citizen1_context = ConversationContext(
            messages=[], 
            system_message_added=False,
            active_user_profile=None
        )
        st.session_state.citizen1_input_key = 0
    
    if 'citizen2_context' not in st.session_state:
        st.session_state.citizen2_context = ConversationContext(
            messages=[],
            system_message_added=False,
            active_user_profile=None
        )
        st.session_state.citizen2_input_key = 0
    
    # Load components
    try:
        embeddings_manager, query_engine, conversation_manager = initialize_components()
        
        # Check if collection is empty and show appropriate message
        collection = embeddings_manager.get_collection()
        if collection.count() == 0:
            st.warning("No documents have been added to the system yet. The chatbot will have limited functionality until documents are added.")
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return
    
    # Load user profiles
    user_profiles = load_user_profiles()
    
    # ===== PAGE LAYOUT =====
    st.markdown("<h1 style='text-align: center;'>Obi</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666666;'>Massachusetts RMV Service Agent</h3>", unsafe_allow_html=True)
    
    # Create two columns for the chat interfaces
    col1, col2 = st.columns(2)
    
    # Citizen 1 Column
    with col1:
        st.markdown("<h3 style='text-align: center;'>Citizen 1</h3>", unsafe_allow_html=True)
        if user_profiles["users"]:
            display_user_info(user_profiles["users"][0])
        if st.button("Start Citizen 1", use_container_width=True):
            if user_profiles["users"]:
                st.session_state.citizen1_context.active_user_profile = user_profiles["users"][0]
                st.session_state.citizen1_context.messages = []
                st.session_state.citizen1_context.system_message_added = False
                process_user_message("Hello?", conversation_manager, st.session_state.citizen1_context, visible=False)
                st.rerun()
        
        # Citizen 1's chat container
        chat1_container = st.container()
        with chat1_container:
            display_chat_messages(st.session_state.citizen1_context.messages)
            
            if st.session_state.citizen1_context.active_user_profile:
                key = f"chat_input_1_{st.session_state.citizen1_input_key}"
                if prompt := st.chat_input("Type your message here...", key=key):
                    if process_user_message(prompt, conversation_manager, st.session_state.citizen1_context):
                        st.session_state.citizen1_input_key += 1
                        st.rerun()
    
    # Citizen 2 Column
    with col2:
        st.markdown("<h3 style='text-align: center;'>Citizen 2</h3>", unsafe_allow_html=True)
        if len(user_profiles["users"]) > 1:
            display_user_info(user_profiles["users"][1])
        if st.button("Start Citizen 2", use_container_width=True):
            if len(user_profiles["users"]) > 1:
                st.session_state.citizen2_context.active_user_profile = user_profiles["users"][1]
                st.session_state.citizen2_context.messages = []
                st.session_state.citizen2_context.system_message_added = False
                process_user_message("Hello?", conversation_manager, st.session_state.citizen2_context, visible=False)
                st.rerun()
        
        # Citizen 2's chat container
        chat2_container = st.container()
        with chat2_container:
            display_chat_messages(st.session_state.citizen2_context.messages)
            
            if st.session_state.citizen2_context.active_user_profile:
                key = f"chat_input_2_{st.session_state.citizen2_input_key}"
                if prompt := st.chat_input("Type your message here...", key=key):
                    if process_user_message(prompt, conversation_manager, st.session_state.citizen2_context):
                        st.session_state.citizen2_input_key += 1
                        st.rerun()

if __name__ == "__main__":
    main()
