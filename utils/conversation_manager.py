from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import anthropic
from .query_engine import QueryEngine, QueryResult
from .chat_storage import ChatStorage
import logging
import time
import re
from datetime import datetime
import uuid

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
    thread_id: str = ""

    def __post_init__(self):
        """Ensure thread_id is set."""
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())

class ConversationManager:
    def __init__(self, query_engine: QueryEngine, api_key: str, chat_storage: Optional[ChatStorage] = None):
        """Initialize conversation manager with query engine, Anthropic credentials, and optional chat storage."""
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")
        self.query_engine = query_engine
        self.client = anthropic.Anthropic(api_key=api_key)
        self.chat_storage = chat_storage
        
        # Core system prompt for Obi's behavior
        self.system_prompt: str = """You are a professional guide helping Massachusetts citizens renew their driver's licenses. Adapt your approach based on user profiles: warm and methodical for detail-oriented users, crisp and efficient for time-sensitive users. NEVER use exclamation points. Use natural questions to guide the conversation forward, ensuring they flow from the discussion rather than feeling tacked on. Your goal is to guide effectively, matching each user's preferred communication style.
        
        INITIAL CONTACT GUIDELINES

        1. Always keep the first response under 50 words and end with a question. 

        2. Assess "bagman_description" immediately for communication preferences:

        3. First response must establish the following and be grounded in "bagman_description" insights:
            - Appropriate formality level
            - Tone
            - Recognition of immediate needs
            - Clear next step
            - Brief qualifying question

        4. NEVER give a numbered list or bullet list in the first response.
        
        INFORMATION HANDLING:
        
        1. Available Information:
            - State it confidently.
            - Adjust context based on user profile details, especially "bagman_description":
                - ie, Detail-oriented users: Provide supporting context and explanations
                - ie, Efficiency-focused users: State essential facts only
            - Connect related information to the user's situation or goal.

        2. Partially Available Information:
            - Share what you know.
            - Tailor verification approach:
                - ie, Detail-oriented users: Offer to help research and explain verification process
                - ie, Efficiency-focused users: Provide direct resource links with minimal explanation

        3. Unavailable Information:
            - Acknowledge limitations transparently.
            - Focus on next steps.
            - Profile-based resource sharing, with a priority given to "bagman_description" insights:
                - ie, Detail-oriented users: Explain available resource options
                - ie, Efficiency-focused users: Share single best resource

        4. Complex Scenarios:
            - Collaborate with users by providing step-by-step guidance and connecting details from different sections when necessary.
            - Guide users to official verification when necessary.

        TONE AND STYLE:
        1. Never use exclamation points. Maintain a calm, professional tone that conveys confidence without excessive enthusiasm.
        2. Adjust formality based on user profile, with a priority given to "bagman_description" insights.
        3. Acknowledge user effort by describing their actions in a straightforward and professional manner, focusing on what they've done or are ready to do without overly praising or labeling behavior (e.g., avoid terms like "proactive").
        4. Empathize with challenges based on user input, but avoid over-empathizing. For users who may value reassurance, offer calm and supportive guidance. For users who prefer efficiency, briefly acknowledge obstacles and move quickly to actionable solutions.
        5. Avoid excessive praise, but offer practical encouragement to build confidence and keep users engaged.
        6. Adjust the pacing and level of detail based on user preferences, with a priority given to "bagman_description" insights:

        BEHAVIORAL GUIDANCE:
        1. Use document information confidently when available.
        2. Synthesize related information into one clear, actionable step at a time.
        3. Frame solutions in user-specific terms that align with the user's needs and preferences, with a priority given to "bagman_description" insights.
        4. Recommend helpful actions (e.g., scheduling appointments or gathering documents). Adapt recommendations to user preferences and personality traits. 
        5. Present information for confirmation when needed.
        6. If users express frustration or confusion, immediately switch to one-clear-step-at-a-time guidance.
        7. Ensure accessibility for users with disabilities or special needs.
        9. Log unresolved queries for future improvements."""
    
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
        
        # Add line breaks before questions at the end
        text = re.sub(r'([.!?])\s*(Which|What|How|Would|Could|Can|Do|Does|Is|Are|Should|Will|Where|When)\s', r'\1\n\n\2 ', text)
        
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
    
    def _create_prompt(self, messages: List[Message], current_query: str, doc_context: str) -> List[Dict[str, str]]:
        """Create the complete prompt including user profile context if available."""
        formatted_messages = []
        
        # Add conversation history with explicit roles, excluding system messages
        for msg in messages:
            if msg.role != "system" and msg.visible:
                formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Always include the current query as a user message with document context
        formatted_messages.append({
            "role": "user",
            "content": f"{current_query}\n\nRelevant Document Context:\n{doc_context}"
        })
        
        return formatted_messages
    
    def _get_enhanced_system_prompt(self, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Get system prompt enhanced with user profile if available."""
        if not user_profile:
            return self.system_prompt
            
        # Extract name preference and bagman description
        bagman_info = user_profile.get('metadata', {}).get('bagman_description', '')
        name_to_use = user_profile['personal']['full_name']
        
        # Log the bagman description for debugging
        logger.info(f"Bagman description for {name_to_use}: {bagman_info}")
        
        if not bagman_info:
            logger.warning(f"No bagman description found for user {name_to_use}")
        
        # Calculate age from date of birth
        age = self._calculate_age(user_profile['personal']['dob'])
        
        # Put user profile information at the beginning for prominence
        profile_context = f"""MANDATORY COMMUNICATION REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:

{bagman_info}

CRITICAL USER DETAILS - REFERENCE THESE IN YOUR RESPONSES:
- Name: {name_to_use}
- Age: {age}
- License Expiration: {user_profile['license']['current']['expiration']}
- Primary Language: {user_profile['personal']['primary_language']}

IMPORTANT LICENSE INFORMATION:
Type: {user_profile['license']['current']['type']}
Number: {user_profile['license']['current']['number']}
Address: {user_profile['addresses']['residential']['street']}, {user_profile['addresses']['residential']['city']}, {user_profile['addresses']['residential']['state']} {user_profile['addresses']['residential']['zip']}

YOU MUST ADAPT YOUR COMMUNICATION STYLE TO MATCH THE USER'S PREFERENCES ABOVE.
YOU MUST USE THE NAME PREFERENCE SPECIFIED IN THE COMMUNICATION REQUIREMENTS.
YOU MUST MAINTAIN THIS STYLE CONSISTENTLY THROUGHOUT THE CONVERSATION.

CORE SYSTEM INSTRUCTIONS:

{self.system_prompt}"""
        
        return profile_context
    
    def get_response(self, query: str, context: ConversationContext, visible: bool = True) -> str:
        """Generate a response using Claude based on query and conversation context."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Add system message if not already added
        if not context.system_message_added:
            context.messages.append(Message(role="system", content=self.system_prompt))
            context.system_message_added = True
        
        # Retrieve relevant documents
        query_results = self.query_engine.query(query)
        context.last_query_results = query_results
        
        # Format document context
        doc_context = self._format_context(query_results)
        
        # Create messages array for Claude
        messages = self._create_prompt(context.messages, query, doc_context)
        
        # Get enhanced system prompt
        system = self._get_enhanced_system_prompt(context.active_user_profile)
        
        # Log the complete system prompt for debugging
        logger.info(f"Complete system prompt:\n{system}")
        
        # Try Claude 3.5 Sonnet first, fall back to Claude 3 Opus if overloaded
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                system=system,
                messages=messages
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
                    system=system,
                    messages=messages
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
        
        # Add user message and assistant response to context
        context.messages.append(Message(role="user", content=query, visible=visible))
        context.messages.append(Message(role="assistant", content=generated_response))
        
        # Save chat thread to storage if available
        if self.chat_storage:
            messages_for_storage = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "visible": msg.visible
                }
                for msg in context.messages
                if msg.role != "system"  # Exclude system messages from storage
            ]
            
            try:
                if context.messages:  # Only save if there are messages
                    self.chat_storage.update_thread(context.thread_id, messages_for_storage)
            except Exception as e:
                logger.error(f"Failed to save chat thread: {str(e)}")
                # Continue with response even if storage fails
        
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
