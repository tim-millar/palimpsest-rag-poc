# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

RUN mkdir -p /app/.streamlit

# Copy dependencies
COPY requirements.txt .
COPY requirements-lock.txt .

# Install system deps (optional: minimal build utils for some packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y && apt-get clean

# Copy application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
