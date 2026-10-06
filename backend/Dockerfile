FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Backend deps
COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

# ml/ is imported by the backend (shared preprocessing/model code), so its
# dependencies must be present too.
COPY ml/requirements.txt /app/ml-requirements.txt
RUN pip install --no-cache-dir -r /app/ml-requirements.txt

# Copy the whole repo so `ml` and `backend/app` are both importable as
# top-level packages (PYTHONPATH=/app).
COPY backend /app/backend
COPY ml /app/ml

ENV PYTHONPATH=/app:/app/backend
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
