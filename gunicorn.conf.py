# Gunicorn configuration file for production FastAPI deployment
# https://docs.gunicorn.org/en/stable/configure.html#configuration-file
# https://docs.gunicorn.org/en/stable/settings.html

import multiprocessing

max_requests = 1000 # Restart worker after N requests
max_requests_jitter = 50 # Add randomness to restart timing
log_file = "-" # Send logs to stdout
bind = "0.0.0.0:80" # Listen on all interfaces, port 80
worker_class = "uvicorn.workers.UvicornWorker" # Async-capable workers for FastAPI
WORKERS_PER_CORE = 2 # Multiplier for I/O-bound applications
WORKER_BUFFER = 1 # Extra worker for handling load spikes
workers = multiprocessing.cpu_count() * WORKERS_PER_CORE + WORKER_BUFFER # number of uvicorn instances