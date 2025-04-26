import os
import json
import base64
import time
import uuid
from typing import Dict, List, Any, Optional, Union
import httpx
import asyncio
import logging
import traceback

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
# Removed LangSmith imports to prevent parent_run_id errors
# from langsmith import Client
# from langsmith import traceable
# from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger('email_agent')
logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

# Configure API keys and service URLs
# Load environment variables if not already loaded
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL", "http://localhost:3000/email")
logger.info(f"Using email service at: {EMAIL_SERVICE_URL}")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "foundmate-email-assistant")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Disabled LangSmith client initialization to prevent parent_run_id attribute errors
langsmith_client = None

# Email service client
async def email_service_request(method, path, access_token, params=None, data=None):
    """Make a request to the email service"""
    url = f"{EMAIL_SERVICE_URL}{path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    logger.debug(f"Making {method.upper()} request to {url}")
    logger.debug(f"Request params: {params}")
    if data:
        logger.debug(f"Request data: {json.dumps(data)[:100]}...")
    
    try:
        async with httpx.AsyncClient() as client:
            if method.lower() == "get":
                response = await client.get(url, headers=headers, params=params)
            elif method.lower() == "post":
                response = await client.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code >= 400:
                error_msg = f"Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            response_data = response.json()
            logger.debug(f"Response data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
            return response_data
            
    except Exception as e:
        logger.error(f"Error in email_service_request: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Tool definitions for the agent
class SearchEmailsInput(BaseModel):
    query: str = Field(description="Search query to find emails")
    labelId: str = Field(default="INBOX", description="Label ID to search within (INBOX, STARRED, SENT, DRAFT)")
    maxResults: int = Field(default=10, description="Maximum number of results to return")

@tool
async def search_emails(access_token: str, search_input: SearchEmailsInput) -> str:
    """Search for emails based on a query and label."""
    try:
        params = {
            "labelId": search_input.labelId,
            "q": search_input.query,
            "maxResults": search_input.maxResults
        }
        result = await email_service_request("get", "/email", access_token, params=params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching emails: {str(e)}")
        return f"Error searching emails: {str(e)}"

class GetEmailInput(BaseModel):
    email_id: str = Field(description="ID of the email to retrieve")

@tool
async def get_email(access_token: str, input_data: GetEmailInput) -> str:
    """Get a specific email by ID."""
    try:
        result = await email_service_request("get", f"/get-message/{input_data.email_id}", access_token)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting email: {str(e)}")
        return f"Error getting email: {str(e)}"

class GetAttachmentInput(BaseModel):
    message_id: str = Field(description="ID of the email")
    attachment_id: str = Field(description="ID of the attachment")

@tool
async def get_attachment(access_token: str, input_data: GetAttachmentInput) -> str:
    """Get a specific email attachment."""
    try:
        result = await email_service_request(
            "get", 
            f"/get-attachment/{input_data.message_id}/{input_data.attachment_id}", 
            access_token
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting attachment: {str(e)}")
        return f"Error getting attachment: {str(e)}"

@tool
async def send_email(access_token: str, to: str, subject: str, body: str, isDraft: bool = False, cc: str = "", bcc: str = "") -> str:
    """Send an email using the Gmail API.
    
    Args:
        access_token: The OAuth access token for Gmail API
        to: Recipient email address
        subject: Email subject
        body: Email body (HTML supported)
        isDraft: Whether to create a draft instead of sending
        cc: CC recipients (comma separated)
        bcc: BCC recipients (comma separated)
        
    Returns:
        Response from the email service
    """
    try:
        data = {
            "to": to,
            "subject": subject,
            "body": body,
            "isDraft": isDraft
        }
        
        if cc:
            data["cc"] = cc
        if bcc:
            data["bcc"] = bcc
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EMAIL_SERVICE_URL + "/send",
                json=data,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return f"Email {'drafted' if isDraft else 'sent'} successfully to {to}"
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return f"Error sending email: {str(e)}"

@tool
async def summarize_email_thread(access_token: str, thread_id: str) -> str:
    """Summarize an email thread."""
    try:
        result = await email_service_request("get", f"/get-thread/{thread_id}", access_token)
        
        # Extract email content from thread
        emails = result.get("messages", [])
        
        if not emails:
            return "Thread contains no messages."
        
        # Prepare content for summarization
        email_contents = []
        for email in emails:
            from_address = email.get("from", "Unknown")
            subject = email.get("subject", "No Subject")
            snippet = email.get("snippet", "")
            date = email.get("date", "")
            
            email_contents.append(f"From: {from_address}\nDate: {date}\nSubject: {subject}\nContent: {snippet}\n")
        
        # Create a prompt for summarization
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        messages = [
            SystemMessage(content="You are an assistant that summarizes email threads concisely."),
            HumanMessage(content=f"Summarize this email thread in a few sentences, highlighting key points and action items:\n\n{('\n---\n').join(email_contents)}")
        ]
        
        summary = llm.invoke(messages).content
        return summary
    except Exception as e:
        logger.error(f"Error summarizing email thread: {str(e)}")
        return f"Error summarizing email thread: {str(e)}"

# Process attachments
# Removed @traceable decorator to fix parent_run_id errors
async def process_attachments(attachment_data):
    """Process different types of attachments and extract content"""
    try:
        if not attachment_data or "content" not in attachment_data:
            return "No content available in attachment"
        
        content = attachment_data["content"]
        mime_type = attachment_data.get("mimeType", "")
        filename = attachment_data.get("filename", "")
        
        # Decode base64 content
        if isinstance(content, str):
            try:
                decoded_content = base64.b64decode(content)
            except:
                return "Unable to decode attachment content"
        
        # Process based on file type
        if mime_type.startswith("text/"):
            # Text file
            try:
                text_content = decoded_content.decode("utf-8")
                return text_content
            except:
                return "Unable to decode text content"
        
        elif mime_type == "application/pdf":
            # Save PDF temporarily and use PDF loader
            temp_file = f"/tmp/{int(time.time())}_{filename}"
            with open(temp_file, "wb") as f:
                f.write(decoded_content)
                
            try:
                loader = PyPDFLoader(temp_file)
                documents = loader.load()
                content = "\n".join([doc.page_content for doc in documents])
                return content
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                return f"Error processing PDF: {str(e)}"
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        elif mime_type == "application/vnd.ms-excel" or mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Excel file
            temp_file = f"/tmp/{int(time.time())}_{filename}"
            with open(temp_file, "wb") as f:
                f.write(decoded_content)
                
            try:
                loader = UnstructuredExcelLoader(temp_file)
                documents = loader.load()
                content = "\n".join([doc.page_content for doc in documents])
                return content
            except Exception as e:
                logger.error(f"Error processing Excel file: {str(e)}")
                return f"Error processing Excel file: {str(e)}"
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        elif mime_type == "text/csv":
            # CSV file
            temp_file = f"/tmp/{int(time.time())}_{filename}"
            with open(temp_file, "wb") as f:
                f.write(decoded_content)
                
            try:
                loader = CSVLoader(temp_file)
                documents = loader.load()
                content = "\n".join([doc.page_content for doc in documents])
                return content
            except Exception as e:
                logger.error(f"Error processing CSV file: {str(e)}")
                return f"Error processing CSV file: {str(e)}"
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        else:
            return f"Unsupported attachment type: {mime_type}"
            
    except Exception as e:
        logger.error(f"Error processing attachment: {str(e)}")
        return f"Error processing attachment: {str(e)}"

# Email summarization function
# Removed @traceable decorator to fix parent_run_id errors
async def summarize_emails(emails_data):
    """Summarize a batch of emails"""
    try:
        if not emails_data or len(emails_data) == 0:
            return "No emails to summarize"
        
        # Prepare email content for summarization
        email_contents = []
        for email in emails_data:
            from_address = email.get("from", "Unknown")
            to_address = email.get("to", "Unknown")
            subject = email.get("subject", "No Subject")
            snippet = email.get("snippet", "")
            date = email.get("date", "")
            
            email_contents.append(f"From: {from_address}\nTo: {to_address}\nDate: {date}\nSubject: {subject}\nContent: {snippet}\n")
        
        # Create a prompt for summarization
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        messages = [
            SystemMessage(content="You are an assistant that summarizes emails concisely, grouping them by topic and highlighting important information."),
            HumanMessage(content=f"Summarize these {len(emails_data)} emails into key topics and insights:\n\n{'\n---\n'.join(email_contents)}")
        ]
        
        summary = llm.invoke(messages).content
        return summary
    except Exception as e:
        logger.error(f"Error summarizing emails: {str(e)}")
        return f"Error summarizing emails: {str(e)}"

# Extract insights from emails
# Removed @traceable decorator to fix parent_run_id errors
async def extract_insights(emails_data):
    """Extract important insights, tasks, and follow-ups from emails"""
    try:
        if not emails_data or len(emails_data) == 0:
            return "No emails to analyze"
        
        # Prepare email content for analysis
        email_contents = []
        for email in emails_data:
            from_address = email.get("from", "Unknown")
            subject = email.get("subject", "No Subject")
            snippet = email.get("snippet", "")
            date = email.get("date", "")
            
            email_contents.append(f"From: {from_address}\nDate: {date}\nSubject: {subject}\nContent: {snippet}\n")
        
        # Create a prompt for insight extraction
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        messages = [
            SystemMessage(content="""You are an assistant that analyzes emails and extracts important insights. Look for:
1. Action items and tasks that need to be completed
2. Follow-ups required
3. Important deadlines or dates
4. Key people mentioned
5. Critical information or updates
6. Questions that need answers

Organize your response by category."""),
            HumanMessage(content=f"Extract insights from these emails:\n\n{'\n---\n'.join(email_contents)}")
        ]
        
        insights = llm.invoke(messages).content
        return insights
    except Exception as e:
        logger.error(f"Error extracting insights: {str(e)}")
        return f"Error extracting insights: {str(e)}"

# The main email processing agent class
class EmailProcessingAgent:
    def __init__(self, access_token: str):
        """Initialize the agent with the email service access token."""
        logger.info("Initializing EmailProcessingAgent")
        self.access_token = access_token
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set. Please add it to your .env file.")
            # Instead of failing immediately, we'll try to load it from .env
            load_dotenv()
            
        # Initialize the LLM with error handling
        logger.info("Initializing ChatOpenAI model")
        try:
            self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
            logger.info("ChatOpenAI model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            logger.error(traceback.format_exc())
            # Using a default fallback if needed
            self.llm = None
        
        # Disable LangChain tracing to prevent parent_run_id attribute errors
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled to avoid parent_run_id errors")
        
        # Create and bind tools to methods to handle the async operations correctly
        self.search_emails_wrapper = lambda search_input: self._wrap_async_tool(search_emails, search_input)
        self.get_email_wrapper = lambda input_data: self._wrap_async_tool(get_email, input_data)
        self.get_attachment_wrapper = lambda input_data: self._wrap_async_tool(get_attachment, input_data)
        self.send_email_wrapper = lambda input_data: self._wrap_async_tool(send_email, input_data)
        self.summarize_thread_wrapper = lambda thread_id: self._wrap_async_tool(summarize_email_thread, thread_id)
        
        # Define properly structured tools for the agent with appropriate decorators
        from langchain.tools import Tool

        # Define synchronous functions that wrap the async ones
        def _search_emails_sync(search_input_str: str) -> str:
            """Search for emails based on a query and label."""
            import json
            import asyncio
            try:
                # Parse the input string as JSON
                search_input_dict = json.loads(search_input_str)
                search_input = SearchEmailsInput(**search_input_dict)
                return asyncio.run(search_emails(self.access_token, search_input))
            except Exception as e:
                logger.error(f"Error in search_emails: {str(e)}")
                return f"Error in search_emails: {str(e)}"
            
        def _get_email_sync(email_id: str) -> str:
            """Get a specific email by ID."""
            import asyncio
            try:
                input_data = GetEmailInput(email_id=email_id)
                return asyncio.run(get_email(self.access_token, input_data))
            except Exception as e:
                logger.error(f"Error in get_email: {str(e)}")
                return f"Error in get_email: {str(e)}"
            
        def _get_attachment_sync(attachment_info: str) -> str:
            """Get a specific email attachment. Format: 'message_id:attachment_id'"""
            import asyncio
            try:
                message_id, attachment_id = attachment_info.split(":")
                input_data = GetAttachmentInput(message_id=message_id, attachment_id=attachment_id)
                return asyncio.run(get_attachment(self.access_token, input_data))
            except Exception as e:
                logger.error(f"Error in get_attachment: {str(e)}")
                return f"Error in get_attachment: {str(e)}"
            
        def _send_email_sync(email_data: str) -> str:
            """Send an email or create a draft."""
            import json
            import asyncio
            try:
                email_dict = json.loads(email_data)
                to = email_dict.get('to', '')
                subject = email_dict.get('subject', '')
                body = email_dict.get('body', '')
                is_draft = email_dict.get('isDraft', False)
                cc = email_dict.get('cc', '')
                bcc = email_dict.get('bcc', '')
                return asyncio.run(send_email(self.access_token, to, subject, body, is_draft, cc, bcc))
            except Exception as e:
                logger.error(f"Error in send_email: {str(e)}")
                return f"Error in send_email: {str(e)}"
            
        def _summarize_thread_sync(thread_id: str) -> str:
            """Summarize an email thread."""
            import asyncio
            try:
                return asyncio.run(summarize_email_thread(self.access_token, thread_id))
            except Exception as e:
                logger.error(f"Error in summarize_thread: {str(e)}")
                return f"Error in summarize_thread: {str(e)}"
        
        # Define our custom tools focused solely on our email-service API
        logger.info("Setting up custom email service tools")
        self.tools = [
            Tool(
                name="search_emails",
                func=_search_emails_sync,
                description="Search for emails based on a query and label. Input should be a JSON string with 'query', 'labelId' (optional), and 'maxResults' (optional) fields."
            ),
            Tool(
                name="get_email",
                func=_get_email_sync,
                description="Get a specific email by ID. Input should be the email ID string."
            ),
            Tool(
                name="get_attachment",
                func=_get_attachment_sync,
                description="Get a specific email attachment. Input should be in the format 'message_id:attachment_id'."
            ),
            Tool(
                name="send_email",
                func=_send_email_sync,
                description="Send an email or create a draft. Input should be a JSON string with 'to', 'subject', 'body', 'cc' (optional), 'bcc' (optional), and 'isDraft' (optional) fields."
            ),
            Tool(
                name="summarize_thread",
                func=_summarize_thread_sync,
                description="Summarize an email thread. Input should be the thread ID string."
            )
        ]
        logger.debug(f"Created {len(self.tools)} custom tools")
        
        # Create a very simple prompt template that avoids format issues
        system_prompt = """You are an AI assistant that helps users manage their emails.
        You have access to tools that can search for emails, get specific emails, get attachments, summarize threads, and send emails.
        Be helpful, concise, and focus on answering the user's query about their emails."""
        
        # We're using a simpler prompt approach to avoid template issues
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Initialize agent with safer approach and better error handling
        logger.info("Creating agent with LangChain")
        try:
            if self.llm is None:
                raise ValueError("LLM is not initialized, cannot create agent")
            
            # Use create_openai_functions_agent which tends to be more reliable
            agent = create_openai_functions_agent(self.llm, self.tools, prompt)
            logger.info("Agent created successfully with OpenAI Functions format")
            
            # Create the agent executor with verbose mode for debugging
            try:
                self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
                logger.info("Agent executor created successfully")
            except Exception as e:
                logger.error(f"Error creating agent executor: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to initialize agent: {str(e)}")
    
    # Removed @traceable decorator to fix parent_run_id errors
    # Helper method to wrap async function calls
    def _wrap_async_tool(self, func, *args, **kwargs):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(self.access_token, *args, **kwargs))
        except Exception as e:
            logger.error(f"Error in _wrap_async_tool: {str(e)}")
            raise
        finally:
            loop.close()
    
    # Removed @traceable decorator to fix parent_run_id errors
    async def process_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a natural language query with context"""
        logger.info(f"Processing query: {context.get('query', 'No query')}")
        try:
            # Check if OpenAI API key is valid
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
                return {
                    "answer": "Error: OpenAI API key is not configured. Please add it to your .env file.",
                    "additional_data": {"error": "Missing API key"}
                }
                
            # Check if agent and llm were properly initialized 
            if not hasattr(self, "agent_executor") or not hasattr(self, "llm") or self.llm is None:
                logger.error("Agent or LLM not properly initialized")
                return {
                    "answer": "Error: The AI system is not properly initialized. Please check your configuration.",
                    "additional_data": {"error": "Initialization failure"}
                }
            
            query = context.get("query", "")
            email_data = context.get("emails", {})
            thread_data = context.get("thread", {})
            
            # Log detailed information about the context
            logger.debug(f"Query: {query}")
            logger.debug(f"Email data type: {type(email_data)}")
            if isinstance(email_data, dict):
                logger.debug(f"Email data keys: {list(email_data.keys())}")
                if 'messages' in email_data:
                    logger.debug(f"Number of messages: {len(email_data.get('messages', []))}")
            
            # Prepare email context as a string to avoid serialization issues
            email_context = ""
            if isinstance(email_data, dict) and 'messages' in email_data:
                messages = email_data.get('messages', [])
                if messages:
                    logger.info(f"Processing {len(messages)} messages in context")
                    email_context += f"You have {len(messages)} emails. Here is a summary:\n"
                    
                    for i, msg in enumerate(messages[:10]):  # Limit to first 10 for brevity
                        from_addr = msg.get('from', 'Unknown')
                        subject = msg.get('subject', 'No Subject')
                        snippet = msg.get('snippet', '')
                        date = msg.get('date', '')
                        email_context += f"Email {i+1}: From {from_addr} | Date: {date} | Subject: {subject} | Snippet: {snippet}\n\n"
            
            # Build a simple, clean input for the agent
            input_text = f"{query}"
            
            # Only add email context if we have any
            if email_context:
                input_text += f"\n\nHere is information about your emails:\n{email_context}"
            
            # Prepare the final input data
            input_data = {
                "input": input_text
            }
            
            logger.debug(f"Final input to agent: {input_text[:100]}...")
            
            logger.info(f"Invoking agent with input: {query}")
            
            # Use a direct try-except block to handle agent execution
            try:
                # Execute in a thread to avoid blocking
                def _run_agent():
                    try:
                        return self.agent_executor.invoke(input_data)
                    except Exception as e:
                        logger.error(f"Agent execution error: {str(e)}")
                        logger.error(traceback.format_exc())
                        return {"output": f"Error during processing: {str(e)}"}
                
                result = await asyncio.to_thread(_run_agent)
                logger.info(f"Agent execution completed with result type: {type(result)}")
                
                # Handle result - ensure we have a proper output
                if isinstance(result, dict) and "output" in result:
                    answer = result["output"]
                    logger.info(f"Answer found in result: {answer[:100]}...")
                else:
                    logger.warning(f"Unexpected result format: {result}")
                    answer = "I encountered an issue processing your request. Please try again."
                
                return {
                    "answer": answer,
                    "additional_data": {}
                }
                
            except Exception as e:
                logger.error(f"Error in agent thread execution: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "answer": f"I'm sorry, I encountered an error processing your request: {str(e)}",
                    "additional_data": {"error": str(e)}
                }
            
        except Exception as e:
            # Handle any errors during processing
            logger.error(f"Error in process_query: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error processing your query: {str(e)}",
                "additional_data": {"error": str(e)}
            }    
            
    async def process_query_original(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a natural language query with context"""
        logger.info(f"Processing query: {context.get('query', 'No query')}")
        try:
            query = context.get("query", "")
            
            if "emails" in context:
                # Process query with specific emails
                emails = context["emails"]
                email_summaries = []
                
                for email in emails:
                    from_address = email.get("from", "Unknown")
                    subject = email.get("subject", "No Subject")
                    snippet = email.get("snippet", "")
                    
                    email_summaries.append(f"From: {from_address}, Subject: {subject}, Content: {snippet}")
                
                augmented_query = f"{query}\n\nHere are the relevant emails:\n{'\n'.join(email_summaries)}"
                response = await self.agent_executor.ainvoke({"input": augmented_query})
                
                # Add additional processing for insights if needed
                if "summarize" in query.lower() or "summary" in query.lower():
                    summary = await summarize_emails(emails)
                    response["additional_data"] = {"summary": summary}
                elif "insight" in query.lower() or "extract" in query.lower() or "analyze" in query.lower():
                    insights = await extract_insights(emails)
                    response["additional_data"] = {"insights": insights}
                
                return response
            
            elif "thread" in context:
                # Process query with a thread
                thread = context["thread"]
                thread_id = thread.get("id", "")
                
                if "summarize" in query.lower() or "summary" in query.lower():
                    summary = await summarize_email_thread(self.access_token, thread_id)
                    return {"answer": summary}
                else:
                    response = await self.agent_executor.ainvoke({"input": query})
                    return response
            
            else:
                # General query without specific context
                try:
                    response = await self.agent_executor.ainvoke({"input": query})
                    return response
                except Exception as e:
                    print(f"Error in agent execution: {str(e)}")
                    # Fallback to simpler approach if the agent execution fails
                    llm_response = await self.llm.ainvoke([SystemMessage(content="You are an AI assistant specialized in email management."), HumanMessage(content=query)])
                    return {"answer": llm_response.content}
                
        except Exception as e:
            return {"answer": f"Error processing query: {str(e)}"}

# Helper function to create an email processing agent
# Removed @traceable decorator to fix parent_run_id errors
def create_email_agent(access_token: str) -> EmailProcessingAgent:
    """Create an email processing agent with the provided access token"""
    trace_id = str(uuid.uuid4())
    os.environ["LANGCHAIN_SESSION"] = f"foundmate-session-{trace_id}"
    return EmailProcessingAgent(access_token)
