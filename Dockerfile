# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_web.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_web.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p web/static/uploads web/database logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=web/app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "web/app.py"]
