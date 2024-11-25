from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import anthropic
from .query_engine import QueryEngine, QueryResult
import logging
import time
import re
from datetime import datetime

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    visible: bool = True

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        return self.role == other.role and self.content == other.content

    def __hash__(self):
        return hash((self.role, self.content))

@dataclass
class ConversationContext:
    messages: List[Message]
    last_query_results: Optional[List[QueryResult]] = None
    system_message_added: bool = False
    active_user_profile: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None

class DummyChatLogger:
    """Fallback chat logger when MongoDB is not available."""
    def __init__(self, *args, **kwargs):
        pass
    
    def start_thread(self) -> str:
        return "local-thread"
    
    def log_message(self, thread_id: str, role: str, content: str) -> None:
        pass

# Try to import ChatLogger, fall back to dummy if not available
try:
    from .chat_logger import ChatLogger
except ImportError:
    logger.warning("ChatLogger not available, using dummy logger")
    ChatLogger = DummyChatLogger

class ConversationManager:
    def __init__(self, query_engine: QueryEngine, api_key: str, mongodb_uri: Optional[str] = None):
        """Initialize conversation manager with query engine and Anthropic credentials."""
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")
        self.query_engine = query_engine
        self.client = anthropic.Anthropic(api_key=api_key)
        self.chat_logger = ChatLogger(mongodb_uri) if mongodb_uri else None
        
        # Core system prompt for Obi's behavior
        self.system_prompt: str = """You are Obi, a professional guide helping Massachusetts citizens renew their driver's licenses. 

KEY GUIDELINES:
1. Adapt your approach based on user profiles: warm and methodical for detail-oriented users, crisp and efficient for time-sensitive users
2. NEVER use exclamation points
3. Ask only ONE question at a time
4. NEVER simulate user responses or create hypothetical dialogue
5. Wait for actual user responses
6. Keep responses focused and direct
7. Guide the conversation naturally, letting questions flow from the discussion

Your goal is to guide effectively while matching each user's preferred communication style."""
    
    def _format_context(self, query_results: List[QueryResult]) -> str:
        """Format retrieved documents into context string."""
        context_parts: List[str] = []
        for result in query_results:
            metadata = result.metadata
            source = str(metadata.get('source', 'Unknown'))
            context_parts.append(f"From {source}:\n{result.text}")
        return "\n\n".join(context_parts)

    def _fix_text_formatting(self, text: str) -> str:
        """Fix essential formatting issues in the text."""
        # Remove space between dollar sign and number
        text = re.sub(r'\$\s+(\d)', r'$\1', text)
        
        # Add spaces between numbers and words
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
        
        # Format bullet points onto separate lines
        text = re.sub(r'([.!?])\s*(•)', r'\1\n\n\2', text)
        text = re.sub(r'([^•])\s*•', r'\1\n\n•', text)
        
        return text

    def _calculate_age(self, dob_str: str) -> int:
        """Calculate age from date of birth string."""
        try:
            dob = datetime.strptime(dob_str, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            return 0
    
    def _create_prompt(self, messages: List[Message], user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Create the complete prompt including user profile context if available."""
        prompt_parts = []
        
        # Add system prompt
        prompt_parts.append(self.system_prompt)
        
        # Add user profile context if available
        if user_profile:
            # Extract name preference from bagman_description if available
            bagman_info = user_profile.get('metadata', {}).get('bagman_description', '')
            name_to_use = user_profile['personal']['full_name']
            
            # Calculate age from date of birth
            age = self._calculate_age(user_profile['personal']['dob'])
            
            # Add explicit instruction about name preference if available
            if bagman_info:
                prompt_parts.append(f"Important Note - {bagman_info}")
            
            profile_context = f"""Current User Profile:
- Name: {name_to_use}
- Age: {age}
- Preferred Language: {user_profile['personal']['primary_language']}
- License Details:
  * Type: {user_profile['license']['current']['type']}
  * Number: {user_profile['license']['current']['number']}
  * Expiration: {user_profile['license']['current']['expiration']}
- Address: {user_profile['addresses']['residential']['street']}, {user_profile['addresses']['residential']['city']}, {user_profile['addresses']['residential']['state']} {user_profile['addresses']['residential']['zip']}
- Additional Notes: {bagman_info}"""
            prompt_parts.append(profile_context)
        
        # Format conversation history as a clear dialogue
        conversation = []
        for msg in messages:
            if msg.role != "system":
                prefix = "User" if msg.role == "user" else "Assistant"
                conversation.append(f"{prefix}: {msg.content}")
        
        if conversation:
            prompt_parts.append("Previous Conversation:\n" + "\n\n".join(conversation))
        
        return "\n\n".join(prompt_parts)
    
    def get_response(self, query: str, context: ConversationContext, visible: bool = True) -> str:
        """Generate a response using Claude based on query and conversation context."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Initialize thread_id if not present and chat_logger is available
        if self.chat_logger and not context.thread_id:
            context.thread_id = self.chat_logger.start_thread()
        
        # Add system message if not already added
        if not context.system_message_added:
            context.messages.append(Message(role="system", content=self.system_prompt))
            context.system_message_added = True
        
        # Add user message
        context.messages.append(Message(role="user", content=query, visible=visible))
        
        # Log user message if chat_logger is available
        if self.chat_logger and context.thread_id:
            self.chat_logger.log_message(context.thread_id, "user", query)
        
        # Retrieve relevant documents
        query_results = self.query_engine.query(query)
        context.last_query_results = query_results
        
        # Format document context
        doc_context = self._format_context(query_results)
        
        # Create complete prompt with document context
        prompt = f"{self._create_prompt(context.messages, context.active_user_profile)}\n\nRelevant Document Context:\n{doc_context}"
        
        # Log the complete prompt for debugging
        logger.info(f"Complete prompt with context:\n{prompt}")
        
        # Try Claude 3.5 Sonnet first, fall back to Claude 3 Opus if overloaded
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            if "overloaded_error" in str(e):
                logging.warning("Claude 3.5 Sonnet is overloaded, falling back to Claude 3 Opus")
                # Add a small delay before retrying with fallback model
                time.sleep(1)
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=500,
                    temperature=0.7,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                raise e
        
        if not hasattr(response, 'content'):
            raise ValueError("No response received from Anthropic")
            
        generated_response = response.content[0].text if response.content else ""
        if not generated_response:
            raise ValueError("Empty response received from Anthropic")
        
        # Fix formatting issues in the response
        generated_response = self._fix_text_formatting(generated_response)
        
        # Add assistant response to context
        context.messages.append(Message(role="assistant", content=generated_response))
        
        # Log assistant response if chat_logger is available
        if self.chat_logger and context.thread_id:
            self.chat_logger.log_message(context.thread_id, "assistant", generated_response)
        
        return generated_response

class SessionManager:
    """Manage conversation sessions in Streamlit."""
    
    @staticmethod
    def initialize_session(st) -> None:
        """Initialize session state variables."""
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = ConversationContext(
                messages=[], 
                system_message_added=False,
                active_user_profile=None
            )
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0
    
    @staticmethod
    def get_conversation_context(st) -> ConversationContext:
        """Retrieve current conversation context."""
        return st.session_state.conversation_context
    
    @staticmethod
    def set_active_user(st, user_profile: Dict[str, Any]) -> None:
        """Set the active user profile for the conversation."""
        st.session_state.conversation_context.active_user_profile = user_profile
        # Reset conversation when switching users
        st.session_state.conversation_context.messages = []
        st.session_state.conversation_context.system_message_added = False
        st.session_state.conversation_context.thread_id = None  # Reset thread_id for new conversation
