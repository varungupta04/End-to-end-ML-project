FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Train the model on startup (generates artifacts/)
RUN python -m src.components.data_ingestion

# Expose Hugging Face's required port
EXPOSE 7860

# Start Flask app
CMD ["python", "app.py"]
