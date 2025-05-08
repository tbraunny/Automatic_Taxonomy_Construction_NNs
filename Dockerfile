# syntax=docker/dockerfile:1
# Base image with Python pre‑installed
FROM python:3.10-slim AS runtime

# Configure working directory inside the image
WORKDIR /app

ENV PATH=".venv/bin:$PATH"
ENV PYTHONPATH="/app"

# ---------------------------------------------------
# Install Python dependencies first (leverages caching)
# ---------------------------------------------------
# Copy only the requirements file at this stage so that
# requirements are re‑installed only when this file changes
COPY requirements.txt ./
COPY requirements/ requirements/

# Upgrade pip and install project requirements
#RUN python -m pip install --upgrade pip && \
#    pip install  -r requirements.txt
RUN apt update
RUN apt install gcc -y
RUN python -m pip install --upgrade pip && pip install uv

RUN uv venv 
RUN uv pip install -r requirements.txt
RUN uv pip install streamlit voila

# ---------------------------
# Copy application source code
# ---------------------------

COPY runapp.sh runapp.sh
COPY src/ src/
COPY utils/ utils/
COPY front_end/ front_end/
COPY data/ data/
COPY .streamlit .streamlit

RUN chmod +x runapp.sh
# If your app exposes a port (uncomment if needed)
# EXPOSE 8000

# Default command — change to your project’s entry‑point
CMD ["streamlit", "run", "src/app.py"]
