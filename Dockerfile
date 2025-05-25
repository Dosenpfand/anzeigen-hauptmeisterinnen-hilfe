# Use an official Python 3.13 runtime as a parent image
FROM python:3.13-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the dependency files
COPY pyproject.toml uv.lock /app/

# Install project dependencies using uv
# Using --system to install into the system site-packages, common for containers
RUN uv sync

# Copy the rest of the application code into the container
COPY ./app.py /app/
COPY ./static /app/static/

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
