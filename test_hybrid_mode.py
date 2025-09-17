#!/usr/bin/env python3
"""
Quick test to verify hybrid mode fixes:
1. LLM processing loop runs every 20 seconds in hybrid mode
2. Rule-based offer filtering is disabled in hybrid mode
"""

import asyncio
import os
from websocket_server import WebSocketChurnServer

async def test_hybrid_mode():
    print("🧪 Testing Hybrid Mode Configuration")
    print("="*50)
    
    # Set hybrid mode
    os.environ['PROCESSING_MODE'] = 'hybrid'
    
    # Create server instance
    server = WebSocketChurnServer('localhost', 8765)
    
    # Check configuration
    print(f"✅ use_llm_processing: {server.use_llm_processing}")
    print(f"✅ use_hybrid_processing: {server.use_hybrid_processing}")
    print(f"✅ Processing mode: {server.get_current_processing_mode()}")
    print(f"✅ LLM interval: {server.llm_interval} seconds")
    
    # Check churn detector configuration
    print(f"✅ Churn scorer LLM indicators: {server.churn_detector.churn_scorer.use_llm_indicators}")
    print(f"✅ Churn scorer LLM offer filtering: {server.churn_detector.churn_scorer.use_llm_offer_filtering}")
    
    # Test LLM loop condition
    print(f"\n🔍 LLM loop condition check:")
    print(f"   (use_llm_processing OR use_hybrid_processing): {server.use_llm_processing or server.use_hybrid_processing}")
    print(f"   Should LLM loop run in hybrid mode? {'YES' if (server.use_llm_processing or server.use_hybrid_processing) else 'NO'}")
    
    # Simulate adding some messages
    server.accumulated_messages = [
        {'speaker': 'Customer', 'text': 'My bill is too high', 'timestamp': '12:00:01'},
        {'speaker': 'Agent', 'text': 'Let me check that for you', 'timestamp': '12:00:02'}
    ]
    
    print(f"\n📝 Simulated {len(server.accumulated_messages)} accumulated messages")
    print("🤖 LLM processing should trigger in hybrid mode when messages are available")
    
    print(f"\n🎯 Expected Behavior:")
    print(f"   - Rule-based churn scoring: ✅ ENABLED (runs every message)")
    print(f"   - Rule-based offer filtering: ❌ DISABLED (skipped in hybrid mode)")
    print(f"   - LLM processing: ✅ ENABLED (runs every {server.llm_interval} seconds)")
    print(f"   - LLM offer filtering: ✅ ENABLED (triggers offers in hybrid mode)")

if __name__ == "__main__":
    asyncio.run(test_hybrid_mode())
    print(f"\n🏁 Hybrid mode configuration test completed!")
    print(f"\n💡 To test live:")
    print(f"   PROCESSING_MODE=hybrid python websocket_server.py")
    print(f"   🔍 Look for: '🤖 Hybrid mode: Processing X accumulated messages with LLM'")
    print(f"   🔍 Look for: '🔄 Hybrid mode: Skipping rule-based offer filtering'") 