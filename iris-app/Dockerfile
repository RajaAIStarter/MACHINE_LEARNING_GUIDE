# Assuming your app file is named iris_app.py and located in the same directory as the Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt to install dependencies
COPY requirements.txt .

# Install the dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory (including your iris_app.py) to the container's /app folder
COPY . .

# Expose the necessary port (assuming the app runs on port 5000)
EXPOSE 5000

# Start the application (update with your actual script name if it's different)
CMD ["python", "app.py"]
