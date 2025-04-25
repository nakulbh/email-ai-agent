import os
import uuid
import logging
import traceback
import sys
import time
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import httpx
import json
from dotenv import load_dotenv

# Load environment variables first, before any other imports
load_dotenv()

# Check for critical environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment variables")
    print("Please run setup_env.py to configure your environment")

# Now import LangChain and other components
from langsmith import Client
from langsmith import traceable

# Configure logging with a timestamp in the filename
log_filename = f'ai_service_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger('ai_service')
logger.info("Starting AI Email Assistant")
logger.info(f"OpenAI API Key configured: {os.getenv('OPENAI_API_KEY') is not None}")

# Log startup information
logger.info(f"Current working directory: {os.getcwd()}")

# Import agent functions after environment is configured
from agent import (
    EmailProcessingAgent, 
    summarize_emails,
    extract_insights
)

# Configure LangSmith if enabled
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "foundmate-email-assistant")

# Initialize LangSmith client if enabled
langsmith_client = None
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    langsmith_client = Client(
        api_url=LANGCHAIN_ENDPOINT,
        api_key=LANGCHAIN_API_KEY,
        project_name=LANGCHAIN_PROJECT
    )

app = FastAPI(title="FoundrMate AI Email Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    access_token: str
    email_ids: Optional[List[str]] = None
    thread_id: Optional[str] = None
    include_attachments: bool = False
    
class EmailResponse(BaseModel):
    response: str
    additional_data: Optional[Dict[str, Any]] = None

# Email service configuration
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL", "http://localhost:3000/email")
logger.info(f"Using email service URL: {EMAIL_SERVICE_URL}")

# Helper functions for email service communication
async def fetch_emails(access_token: str, params: Dict[str, Any] = None):
    """Fetch emails from email service"""
    logger.info(f"Fetching emails with params: {params}")
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {access_token}"}
            logger.debug(f"Making GET request to {EMAIL_SERVICE_URL} with params: {params}")
            response = await client.get(EMAIL_SERVICE_URL, headers=headers, params=params)
            logger.debug(f"Got response with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch emails: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=response.status_code, detail=error_msg)
                
            response_data = response.json()
            logger.info(f"Successfully fetched {len(response_data.get('messages', []))} emails")
            return response_data
    except Exception as e:
        logger.error(f"Error in fetch_emails: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def fetch_email_details(access_token: str, email_id: str):
    """Fetch specific email details including attachments"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get(f"{EMAIL_SERVICE_URL}/{email_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch email details")
        return response.json()

async def fetch_attachments(access_token: str, message_id: str, attachment_id: str):
    """Fetch email attachment"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get(
            f"{EMAIL_SERVICE_URL.replace('/email', '')}/get-attachment/{message_id}/{attachment_id}", 
            headers=headers
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch attachment")
        return response.json()

# Agent instance cache (in production, use Redis or another distributed cache)
agent_cache = {}

async def get_or_create_agent(access_token: str) -> EmailProcessingAgent:
    """Get or create an email agent for the given access token"""
    if access_token not in agent_cache:
        logger.info(f"Creating new EmailProcessingAgent for this session")
        try:
            agent_cache[access_token] = EmailProcessingAgent(access_token)
            logger.info(f"Agent created successfully and cached")
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    else:
        logger.info(f"Using cached EmailProcessingAgent")
    return agent_cache[access_token]

# Routes
@app.get("/")
async def root():
    return {"message": "FoundrMate AI Email Assistant API"}

@app.post("/process-query", response_model=EmailResponse)
@traceable(name="api_process_query")
async def process_query(request: QueryRequest):
    """Process a natural language query about emails"""
    logger.info(f"Processing query: '{request.query}'")
    try:
        # Get or create agent
        logger.debug(f"Getting agent for request")
        agent = await get_or_create_agent(request.access_token)
        logger.debug(f"Agent created/retrieved successfully")
        
        # Prepare context based on query type
        context = {"query": request.query, "access_token": request.access_token}
        logger.debug(f"Prepared base context")
        
        if request.email_ids:
            logger.info(f"Processing specific email IDs: {request.email_ids}")
            # Fetch specific emails
            emails_data = []
            for email_id in request.email_ids:
                logger.debug(f"Fetching details for email ID: {email_id}")
                try:
                    email_data = await fetch_email_details(request.access_token, email_id)
                    logger.debug(f"Successfully fetched email details for ID: {email_id}")
                    
                    if request.include_attachments and "attachments" in email_data:
                        logger.debug(f"Processing {len(email_data.get('attachments', []))} attachments")
                        for attachment in email_data.get("attachments", []):
                            try:
                                logger.debug(f"Fetching attachment: {attachment['id']}")
                                attachment_data = await fetch_attachments(
                                    request.access_token, email_id, attachment["id"]
                                )
                                attachment["content"] = attachment_data.get("content")
                                logger.debug(f"Attachment content retrieved successfully")
                            except Exception as e:
                                logger.error(f"Error fetching attachment {attachment['id']}: {str(e)}")
                                logger.error(traceback.format_exc())
                                
                    emails_data.append(email_data)
                except Exception as e:
                    logger.error(f"Error fetching email details for ID {email_id}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            logger.info(f"Added {len(emails_data)} emails to context")
            context["emails"] = emails_data
        
        elif request.thread_id:
            # Fetch thread
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {request.access_token}"}
                response = await client.get(
                    f"{EMAIL_SERVICE_URL.replace('/email', '')}/get-thread/{request.thread_id}", 
                    headers=headers
                )
                thread_data = response.json()
                context["thread"] = thread_data
        
        else:
            # Fetch emails based on the query
            # Analyze the query to determine search parameters
            logger.info(f"No specific email IDs, analyzing query to determine parameters")
            if "important" in request.query.lower():
                params = {"filter": "important"}
                logger.info(f"Query contains 'important', using filter=important")
            elif "starred" in request.query.lower():
                params = {"labelId": "STARRED"}
                logger.info(f"Query contains 'starred', using labelId=STARRED")
            elif "sent" in request.query.lower():
                params = {"labelId": "SENT"}
                logger.info(f"Query contains 'sent', using labelId=SENT")
            elif "draft" in request.query.lower():
                params = {"labelId": "DRAFT"}
                logger.info(f"Query contains 'draft', using labelId=DRAFT")
            else:
                params = {"labelId": "INBOX"}
                logger.info(f"No specific keywords found, using default labelId=INBOX")
            
            try:    
                logger.debug(f"Fetching emails with params: {params}")
                emails_data = await fetch_emails(request.access_token, params)
                if isinstance(emails_data, dict) and 'messages' in emails_data:
                    logger.info(f"Retrieved {len(emails_data.get('messages', []))} emails")
                    context["emails"] = emails_data
                else:
                    logger.warning(f"Unexpected email data format: {type(emails_data)}")
                    context["emails"] = {"messages": []}
            except Exception as e:
                logger.error(f"Error fetching emails with params {params}: {str(e)}")
                logger.error(traceback.format_exc())
                context["emails"] = {"messages": []}
        
        # Process the query with the agent
        logger.info(f"Sending query to agent for processing with context keys: {list(context.keys())}")
        try:
            response = await agent.process_query(context)
            logger.info(f"Received response from agent")
            logger.debug(f"Agent response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'!r}")
            
            if not isinstance(response, dict) or "answer" not in response:
                logger.error(f"Invalid response format from agent: {response!r}")
                return EmailResponse(
                    response="I encountered an error processing your request. The response format was invalid.",
                    additional_data={"error": "Invalid response format"}
                )
                
            return EmailResponse(
                response=response["answer"],
                additional_data=response.get("additional_data")
            )
        except Exception as e:
            logger.error(f"Error in agent.process_query: {str(e)}")
            logger.error(traceback.format_exc())
            return EmailResponse(
                response=f"I encountered an error processing your request: {str(e)}",
                additional_data={"error": str(e), "traceback": traceback.format_exc()}
            )
        
    except Exception as e:
        logger.error(f"Unhandled exception in process_query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/summarize-emails")
@traceable(name="api_summarize_emails")
async def api_summarize_emails(request: Request):
    """Summarize a batch of emails"""
    data = await request.json()
    access_token = data.get("access_token")
    email_ids = data.get("email_ids", [])
    
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token is required")
        
    if not email_ids:
        raise HTTPException(status_code=400, detail="Email IDs are required")
    
    # Fetch email details
    emails_data = []
    for email_id in email_ids:
        email_data = await fetch_email_details(access_token, email_id)
        emails_data.append(email_data)
        
    # Generate summary
    summary = await summarize_emails(emails_data)
    
    return {"summary": summary}

@app.post("/extract-insights")
@traceable(name="api_extract_insights")
async def api_extract_insights(request: Request):
    """Extract insights from emails"""
    data = await request.json()
    access_token = data.get("access_token")
    email_ids = data.get("email_ids", [])
    
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token is required")
        
    if not email_ids:
        raise HTTPException(status_code=400, detail="Email IDs are required")
    
    # Fetch email details
    emails_data = []
    for email_id in email_ids:
        email_data = await fetch_email_details(access_token, email_id)
        emails_data.append(email_data)
        
    # Extract insights
    insights = await extract_insights(emails_data)
    
    return {"insights": insights}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
