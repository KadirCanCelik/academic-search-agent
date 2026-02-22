#!/bin/bash

# --- START BACKEND ---
# Run the FastAPI server in the background (using the '&' operator)
# It will run on port 8000 locally inside the container
echo "Starting FastAPI Backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# --- START FRONTEND ---
# Run the Streamlit UI in the foreground
# Hugging Face Spaces strictly requires the main application to listen on port 7860
echo "Starting Streamlit Frontend..."
streamlit run app/frontend/ui.py --server.port 7860 --server.address 0.0.0.0