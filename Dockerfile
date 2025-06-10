# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./src /app/src

# Set environment variables for LangChain tracing
ENV LANGCHAIN_TRACING_V2=true
# Note: LANGCHAIN_API_KEY, OPENAI_API_KEY, etc., will be passed from the .env file

# Command to run the application
# We'll set a default command, but this can be overridden.
# For now, we'll just keep the container running.
CMD ["tail", "-f", "/dev/null"] 