FROM python:3.10-slim

# 1. Install system dependencies (needed for shap/sklearn)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory to the HF default
WORKDIR /code

# 3. Copy requirements and install (as root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your folders
# On HF, the default user will own these if we just COPY them
COPY models/ ./models/
COPY src/ ./src/

# 5. Set permissions to be safe
RUN chmod -R 777 /code

# 6. Use the correct module path
# Since app.py is INSIDE src, 'src.app:app' is correct
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]