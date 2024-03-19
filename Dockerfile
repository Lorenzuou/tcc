# Use the same base image as in your docker-compose file
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .


# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt