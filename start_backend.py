#!/usr/bin/env python3
"""
Startup script for the Live Churn Detection WebSocket Server
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import websockets
        import deepgram
        import sounddevice
        import numpy
        import vaderSentiment
        import sentence_transformers
        import transformers
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    print("=== Live Churn Detection Backend ===")
    print("Initializing WebSocket server...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the server
    try:
        from websocket_server import main as run_server
        import asyncio
        
        print("Starting WebSocket server...")
        print("Frontend should connect to: ws://localhost:8765")
        print("Press Ctrl+C to stop the server")
        
        asyncio.run(run_server())
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 