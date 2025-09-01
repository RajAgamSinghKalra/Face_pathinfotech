#!/usr/bin/env python3
"""
start_backend.py - Start the FastAPI backend server for face search
"""

import uvicorn
import sys
from pathlib import Path

if __name__ == "__main__":
    # Check if api_server_simple.py exists
    api_server_path = Path("api_server_simple.py")
    if not api_server_path.exists():
        print("❌ api_server_simple.py not found!")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)

    print("🚀 Starting Face Search API Server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🔧 Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            "api_server_simple:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
