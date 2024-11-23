# OBI - Massachusetts RMV Service Agent

A Streamlit-based conversational AI assistant that helps citizens navigate the driver's license renewal process.

## Features

- Interactive chat interface with context-aware responses
- User profile integration for personalized guidance
- Secure handling of sensitive information
- Support for multiple languages based on user preferences
- Graceful handling of information gaps and uncertainty

## Conversation Guidelines

Obi follows specific guidelines to ensure consistent, helpful interactions:

1. Communication Style:
   - Initial responses are concise (under 75 words)
   - First responses always end with a question
   - Maintains a calm, patient tone without exclamation marks
   - Uses step-by-step guidance when users show uncertainty
   - Addresses users in their preferred language
   - Uses name preferences from user profiles

2. Query Handling:
   - Seeks clarification for unclear or vague queries
   - Provides examples to help users be more specific
   - Confirms critical information (personal details, payments, appointments)
   - Guides users to verify or correct disputed information
   - Offers clear paths for resolving complex issues

3. Information Management:
   - Verifies information against official documentation
   - Only provides details explicitly present in source data
   - Transparently acknowledges when information isn't available
   - Provides actionable next steps for finding missing information
   - Includes relevant contact details and resources
   - Formats appointment times with clear bullet points

4. Support and Accessibility:
   - Ensures accessibility for users with disabilities
   - Provides resources for non-standard scenarios
   - Offers language translation support
   - Creates clear escalation paths for unresolved issues
   - Logs queries for continuous improvement

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
ANTHROPIC_API_KEY=your_api_key_here
MODEL_NAME=all-MiniLM-L6-v2
CHROMA_DB_PATH=./chroma_db
DOCUMENTS_PATH=./data/drivers_license_docs
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
obi-sl/
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── conversation_manager.py
│   ├── embeddings_manager.py
│   └── query_engine.py
├── data/
│   └── drivers_license_docs/  # Document storage
├── chroma_db/           # Vector database storage
├── requirements.txt     # Project dependencies
├── .env                # Environment variables
└── app.py              # Main application
```

## Usage

1. Select a user profile (Citizen 1 or Citizen 2)
2. Start chatting with Obi about your driver's license renewal
3. Follow Obi's guidance and provide requested information
4. Receive personalized assistance throughout the renewal process

## Dependencies

- streamlit
- anthropic
- sentence-transformers
- chromadb
- python-dotenv
- pyyaml
- And more (see requirements.txt)

## Deployment

The application is configured for deployment on Streamlit Cloud. Make sure to:

1. Set up the required secrets in Streamlit Cloud
2. Configure the deployment environment
3. Enable any necessary integrations

## Security

- Sensitive information is handled securely
- No personal data is stored permanently
- Environment variables are used for API keys
- CORS and XSRF protection enabled
