# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements first (this makes future builds faster)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# (Thanks to .dockerignore, it won't copy the bad files!)
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Command to run your app 
CMD ["python", "app.py"]