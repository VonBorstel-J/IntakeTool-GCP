# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc && apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY . .

# Stage 2: Production
FROM python:3.11-slim AS production

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependencies and application code from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Expose application port
EXPOSE 8080

# Healthcheck to ensure the service is running
HEALTHCHECK --interval=30s --timeout=5s CMD curl --fail http://localhost:8080/api/status || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
