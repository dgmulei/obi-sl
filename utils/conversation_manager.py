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

class ConversationManager:
    def __init__(self, query_engine: QueryEngine, api_key: str):
        """Initialize conversation manager with query engine and Anthropic credentials."""
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")
        self.query_engine = query_engine
        self.client = anthropic.Anthropic(api_key=api_key)
        
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
- Additional Notes: {bagman_info}
"""
            prompt_parts.append(profile_context)
        
        # Add conversation history without explicit role prefixes
        for msg in messages:
            if msg.role != "system":
                prompt_parts.append(msg.content)
        
        return "\n\n".join(prompt_parts)
    
    def get_response(self, query: str, context: ConversationContext, visible: bool = True) -> str:
        """Generate a response using Claude based on query and conversation context."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Add system message if not already added
        if not context.system_message_added:
            context.messages.append(Message(role="system", content=self.system_prompt))
            context.system_message_added = True
        
        # Add user message
        context.messages.append(Message(role="user", content=query, visible=visible))
        
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
