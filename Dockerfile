FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Copy .env file specifically (ensure it's not excluded in .dockerignore)
COPY .env /app/.env

# Use bash shell for commands
SHELL ["/bin/bash", "-c"]

# Load environment variables using bash
RUN set -a && . /app/.env && set +a


# Fix syntax error in agent.py with raw string pattern (triple quotes to handle escaping)
RUN sed -i "s/{\'\\n---\\n\'.join(email_contents)}/{('\\n---\\n').join(email_contents)}/g" /app/agent.py

# Expose the port that the application runs on
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["/bin/bash", "-c", ". /app/.env && uvicorn main:app --host 0.0.0.0 --port 5000"]