# Use an official lightweight Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app/src

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure necessary directories exist inside the container
RUN mkdir -p src/logs src/output src/plots


# Set the command to run the main script
CMD ["python", "src/main.py"]
