# FoundrMate AI Email Assistant

AI-powered email assistant for FoundrMate that can process emails, extract insights, summarize content, and handle attachments.

## Features

- Natural language processing of email queries
- Email summarization and insight extraction
- Attachment processing (PDF, Excel, CSV, text files)
- Integration with Gmail API and custom email service
- FastAPI endpoints for easy integration

## Setup

1. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
EMAIL_SERVICE_URL=http://localhost:3000/email
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn main:app --reload --port 5000
```

## API Endpoints

### Process Natural Language Query
```
POST /process-query
```
Body:
```json
{
  "query": "Summarize my recent emails from John",
  "access_token": "your_oauth_token",
  "email_ids": ["optional_specific_email_ids"],
  "thread_id": "optional_thread_id",
  "include_attachments": false
}
```

### Summarize Emails
```
POST /summarize-emails
```
Body:
```json
{
  "access_token": "your_oauth_token",
  "email_ids": ["email_id1", "email_id2"]
}
```

### Extract Insights
```
POST /extract-insights
```
Body:
```json
{
  "access_token": "your_oauth_token",
  "email_ids": ["email_id1", "email_id2"]
}
```

## Example Queries

- "Summarize my last 10 emails"
- "Find all emails about project deadlines"
- "Extract action items from emails in my inbox"
- "Draft a reply to the last email from John about the meeting"
- "What attachments were in the emails from marketing this week?"
