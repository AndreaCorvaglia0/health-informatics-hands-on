# The starting point: a small Linux image with Python 3.12 already installed
FROM python:3.12-slim

# The folder inside the container where the app lives
WORKDIR /app

# Install the packages first: this layer is reused if requirements.txt does not change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, the model, the samples)
COPY . .

# Streamlit listens on port 8501
EXPOSE 8501

# The command that runs when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
