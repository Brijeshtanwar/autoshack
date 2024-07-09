FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY [ "requirements.txt",  "./"]

# Install the required packages
RUN pip install -r requirements.txt

# Copy the model, script, OpenAPI spec and congif.ini files to the container
#  COPY predict.py, src/*, config.ini, open-api-spec.json, model_metadata/*, trained_model/*, ./
COPY src/ ./src/
COPY model_metadata/ ./model_metadata/
COPY trained_model/ ./trained_model/
COPY [ "predict.py", "config.ini", "open-api-spec.json", "./"]

# Expose port 9595 for external access
EXPOSE 9595

# Define the entrypoint for the container
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9595", "--timeout=3600", "predict:app"]