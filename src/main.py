#!/usr/bin/env python3
"""
Main entry point for the AI Study Helper FastAPI application.
"""
import uvicorn
from ai_app.api import app

if __name__ == "__main__":
    uvicorn.run(
        "ai_app.api:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )