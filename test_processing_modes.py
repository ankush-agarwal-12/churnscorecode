#!/usr/bin/env python3
"""
Test script to demonstrate the three processing modes:
1. Rule-based only
2. LLM-only  
3. Hybrid (both rule-based and LLM)

Usage:
PROCESSING_MODE=rule python test_processing_modes.py
PROCESSING_MODE=llm python test_processing_modes.py  
PROCESSING_MODE=hybrid python test_processing_modes.py
"""

import os
import sys
from simple_churn_scorer import SimpleChurnScorer

def test_processing_mode():
    """Test the current processing mode"""
    mode = os.getenv('PROCESSING_MODE', 'rule').lower()
    print(f"🧪 Testing {mode.upper()} processing mode\n")
    
    # Initialize churn scorer
    scorer = SimpleChurnScorer()
    
    # Configure mode
    if mode == 'llm':
        scorer.use_llm_indicators = True
        scorer.use_llm_offer_filtering = True
        print("🤖 Configured for LLM-only processing")
    elif mode == 'hybrid':
        scorer.use_llm_indicators = False  # Rule-based for churn
        scorer.use_llm_offer_filtering = False  # Rule-based for offers
        print("🔄 Configured for Hybrid processing (rule-based base + LLM insights)")
    else:
        scorer.use_llm_indicators = False
        scorer.use_llm_offer_filtering = False
        print("🧠 Configured for Rule-based only processing")
    
    print(f"✅ Mode: {mode}")
    print(f"✅ LLM indicators: {scorer.use_llm_indicators}")
    print(f"✅ LLM offer filtering: {scorer.use_llm_offer_filtering}")
    
    # Test with a sample message
    print("\n" + "="*60)
    print("📝 Testing with sample customer message:")
    print("Customer: 'My bill is $200 again. I barely watch TV. This is too expensive!'")
    print("="*60)
    
    try:
        event = scorer.process_customer_message(
            "My bill is $200 again. I barely watch TV. This is too expensive!",
            "Let me check your account and see what we can do about that."
        )
        
        print(f"\n📊 Results:")
        print(f"   Churn Score: {scorer.current_score:.1f}/100")
        print(f"   Risk Delta: {event.risk_delta:+.1f}")
        print(f"   Sentiment: {event.sentiment_score:.3f}")
        print(f"   Emotion: {event.emotion_result['dominant_emotion']}")
        print(f"   Detected Patterns: {event.detected_patterns}")
        
        # Test offer filtering
        print(f"\n🎁 Testing offer filtering:")
        offers = scorer.get_offers_for_agent("My bill is too high and I barely watch TV")
        
        accepted_offers = [o for o in offers if o.get('accepted', True)]
        rejected_offers = [o for o in offers if not o.get('accepted', True)]
        
        print(f"   ✅ Accepted offers: {len(accepted_offers)}")
        print(f"   ❌ Rejected offers: {len(rejected_offers)}")
        
        if rejected_offers:
            print(f"\n📋 Sample rejection reasons:")
            for i, offer in enumerate(rejected_offers[:3], 1):  # Show first 3
                reason = offer.get('rejection_reason', 'No reason specified')
                print(f"   {i}. {offer['title']}: {reason}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Processing Mode Test Suite")
    print("="*50)
    
    # Show environment variables
    mode = os.getenv('PROCESSING_MODE', 'rule')
    print(f"🔧 Environment: PROCESSING_MODE={mode}")
    
    if len(sys.argv) > 1:
        # Allow command line override
        mode = sys.argv[1].lower()
        os.environ['PROCESSING_MODE'] = mode
        print(f"🔧 Override: PROCESSING_MODE={mode}")
    
    print()
    test_processing_mode()
    
    print(f"\n🏁 Test completed for {mode.upper()} mode!")
    print("\n💡 To test other modes:")
    print("   PROCESSING_MODE=rule python test_processing_modes.py")
    print("   PROCESSING_MODE=llm python test_processing_modes.py") 
    print("   PROCESSING_MODE=hybrid python test_processing_modes.py") 