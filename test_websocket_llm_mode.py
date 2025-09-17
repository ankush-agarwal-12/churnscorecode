#!/usr/bin/env python3

import asyncio
import json
import websockets
import time

async def test_llm_mode():
    """Test LLM mode functionality in the websocket server"""
    
    print("🔍 TESTING WEBSOCKET LLM MODE")
    print("=" * 50)
    
    try:
        # Connect to websocket server
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to websocket server")
            
            # Get initial status
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await websocket.recv()
            status = json.loads(response)
            print(f"📊 Initial Status: {status}")
            
            # Test enabling LLM mode
            print("\n🤖 Enabling LLM processing mode...")
            await websocket.send(json.dumps({
                "type": "set_llm_mode", 
                "enabled": True
            }))
            response = await websocket.recv()
            update = json.loads(response)
            print(f"✅ LLM Mode Update: {update}")
            
            # Get status after enabling LLM
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await websocket.recv()
            status = json.loads(response)
            print(f"📊 Status after LLM enable: {status}")
            
            # Test disabling LLM mode
            print("\n🧠 Disabling LLM processing mode (back to rule-based)...")
            await websocket.send(json.dumps({
                "type": "set_llm_mode", 
                "enabled": False
            }))
            response = await websocket.recv()
            update = json.loads(response)
            print(f"✅ LLM Mode Update: {update}")
            
            # Get final status
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await websocket.recv()
            status = json.loads(response)
            print(f"📊 Final Status: {status}")
            
            print("\n✅ LLM mode test completed successfully!")
            
    except ConnectionRefusedError:
        print("❌ Could not connect to websocket server. Make sure it's running on localhost:8765")
        print("   Run: python websocket_server.py")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    print("🚀 Testing WebSocket LLM Mode Integration")
    print("📝 This test requires the websocket server to be running")
    print("   Start server: python websocket_server.py")
    print("   Or with LLM mode: USE_LLM_PROCESSING=true python websocket_server.py")
    print("-" * 50)
    
    asyncio.run(test_llm_mode()) 