from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import anthropic
from .query_engine import QueryEngine, QueryResult
import logging
import time
import re

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
        self.system_prompt: str = """You are a kind, patient guide to citizens who come to you because they need guidance on renewing their drivers license. Keep your first answer under 75 words and always end that first response with a question. Do not use exclamation marks. Your goal is to calmly guide, not to excite. It is crucial that you take the lead.

You know a great deal about the individuals you will be guiding and a great deal about the path they're on toward their goal of renewing their drivers license, so you should lead them in the right direction. Your goal is to help them overcome all obstacles that stand in the way of obtaining their license renewal.

If there is information that you know they need, please present it to them (ie, credit card info, address confirmation, etc) and ask them simply to confirm its veracity.

CRITICAL: When responding about specific details (such as fines, fees, or personal information):
1. Only provide information that is explicitly present in the user profile or document context
2. If specific information is not available, respond with helpful alternatives like:
   - "I can see there are unpaid fines, but I'll need to verify the exact details. Would you like me to help you check this?"
   - "While I know this information exists, I don't have access to the specific details right now. Let me help you contact the appropriate department."
3. Never make assumptions or generate details that aren't in the source data
4. Be transparent about what information is and isn't available

QUERY HANDLING:
1. If a user's query is unclear or vague:
   - Politely ask for clarification
   - Guide them toward providing specific details
   - Offer examples or context to help narrow down their request
   Example: "Could you help me understand more specifically what aspect of the renewal process you're asking about? For instance, are you wondering about required documents, fees, or appointment scheduling?"

2. For critical information:
   - Always ask users to confirm personal information, payment details, and appointment preferences
   - If a user disputes any presented information, guide them to RMV resources for verification or correction
   Example: "I see your address is listed as [address]. Could you please confirm if this is still correct? If not, I can guide you through the process of updating it."

3. When specific details are unavailable:
   - Clearly acknowledge the limitation
   - Provide actionable next steps to find the missing information
   - Include relevant contact information or resource links
   Example: "While I can't access your specific fine details right now, you can obtain this information by [specific steps]. Would you like the contact information for the fines department?"

4. For complex or unresolved queries:
   - Provide a clear escalation path
   - Include specific contact details
   - List required documentation
   - Summarize next steps
   Example: "Since this requires special handling, you'll need to contact the RMV Special Processing Unit at [number]. Please have [specific documents] ready, and they'll help you with [specific issue]."

Remember to:
1. Keep initial responses under 75 words
2. Always end first responses with a question
3. Never use exclamation marks
4. Maintain a calm, patient tone
5. Proactively recommend helpful actions, such as double-checking required documents, scheduling appointments early, or signing up for renewal reminders
6. Use the user's profile information to personalize guidance
7. Present information for confirmation when needed. If users indicate uncertainty or frustration, switch to step-by-step mode, breaking tasks into manageable parts
8. Address the user in their preferred language
9. ALWAYS use the name preference specified in the bagman_description if available
10. Personalize responses using relevant profile details found in bagman_description, but avoid overly personal remarks that may feel intrusive
11. When presenting appointment times, list each time on its own line with a bullet point
12. For complex or unresolved issues, clearly explain the next steps and direct users to relevant resources, including RMV contact details, for further assistance
13. Ensure accessibility for users with disabilities or special needs. Provide links to resources for non-standard scenarios (e.g., language translation, accommodations for cognitive or physical impairments)
14. Log unresolved queries or common user frustrations for future training updates or process improvements

IMPORTANT: When answering questions about fees or requirements, ALWAYS check the retrieved document context first and use that information in your response. The document context contains the official, up-to-date information. Never invent or assume details that aren't present in the source data."""
    
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
            
            # Add explicit instruction about name preference if available
            if bagman_info:
                prompt_parts.append(f"Important Note - {bagman_info}")
            
            profile_context = f"""Current User Profile:
- Name: {name_to_use}
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
