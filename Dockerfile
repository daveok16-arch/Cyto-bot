FROM python:3.12-slim

# Cloud Run / any container host. Runs one signal cycle per invocation as a job,
# or loops on a schedule if SIGNAL_INTERVAL_SECONDS is set (see src.scheduler).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIGNAL_SYMBOL=GC=F \
    SIGNAL_BAR=5min \
    SIGNAL_HISTORY=5d \
    SIGNAL_PAYOUT=0.80

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/

# never bake credentials into the image; they come from the platform secret store
RUN useradd -m runner && chown -R runner:runner /app
USER runner

CMD ["python", "-m", "src.cloud_run"]