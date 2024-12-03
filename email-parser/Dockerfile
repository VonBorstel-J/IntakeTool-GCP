# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc && apt-get autoremove -y && apt-get clean
# Copy the entire application source code to the container
COPY . .

# Stage 2: Production
FROM python:3.11-slim AS production

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary files from the builder stage
COPY --from=builder /app /app
COPY static /app/static
COPY main.py /app/main.py
COPY config.py /app/config.py
COPY prompts.yaml /app/prompts.yaml

# Use .dockerignore to exclude unnecessary files from being copied into the image
# Ensure sensitive files like .env are excluded

# Expose the application port
EXPOSE 8080

# Healthcheck to ensure the service is running
HEALTHCHECK CMD curl --fail http://localhost:8080/api/status || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
