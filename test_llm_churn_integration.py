#!/usr/bin/env python3

from simple_churn_scorer import SimpleChurnScorer

def test_llm_churn_integration():
    """Test LLM integration with simple churn scorer"""
    
    print("🔍 TESTING LLM INTEGRATION WITH CHURN SCORER")
    print("=" * 70)
    
    # Initialize scorer
    scorer = SimpleChurnScorer()
    
    # Test messages from the conversation
    test_messages = [
        {
            "customer": "I just opened my bill and it's $200 again. Last month it was $180. Why is it going up every time?",
            "agent": "I understand that's frustrating. Let me walk through your bill to identify the changes.",
            "description": "Initial billing complaint"
        },
        {
            "customer": "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for.",
            "agent": "I see here that your promotional discount expired a month ago, and there's a rental charge for a second set-top box.",
            "description": "TV usage + frustration"
        },
        {
            "customer": "Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper.",
            "agent": "I understand. Since you mentioned you barely watch TV, Do you want to remove that?",
            "description": "Competitor mention"
        },
        {
            "customer": "Alright, let's go with that then. Please make sure the new charges reflect next month.",
            "agent": "I've updated your plan and removed the rental. You'll see the changes in your next cycle. Anything else I can help you with today?",
            "description": "Positive resolution"
        }
    ]
    
    print("🧠 RULE-BASED DETECTION (Current System)")
    print("=" * 70)
    
    # Test with rule-based detection (default)
    scorer.use_llm_indicators = False
    scorer.reset_conversation()
    
    for i, test_case in enumerate(test_messages, 1):
        print(f"\n📍 MESSAGE {i}: {test_case['description']}")
        print(f"Customer: {test_case['customer']}")
        print("-" * 50)
        
        event = scorer.process_customer_message(test_case['customer'], test_case['agent'])
        current_score = scorer.get_current_score()
        
        print(f"🎯 Churn Score: {current_score}/100 (Risk Delta: {event.risk_delta:+.1f})")
        print(f"📋 Detected Patterns: {event.detected_patterns}")
    
    print("\n" + "=" * 70)
    print("🤖 LLM-BASED DETECTION (New System)")
    print("=" * 70)
    
    # Test with LLM detection
    scorer.use_llm_indicators = True
    scorer.reset_conversation()
    
    for i, test_case in enumerate(test_messages, 1):
        print(f"\n📍 MESSAGE {i}: {test_case['description']}")
        print(f"Customer: {test_case['customer']}")
        print("-" * 50)
        
        event = scorer.process_customer_message(test_case['customer'], test_case['agent'])
        current_score = scorer.get_current_score()
        
        print(f"🎯 Churn Score: {current_score}/100 (Risk Delta: {event.risk_delta:+.1f})")
        print(f"📋 Detected Patterns: {event.detected_patterns}")
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETE")
    
    # Show final comparison
    print(f"\nℹ️  This test compares rule-based vs LLM-based pattern detection")
    print(f"ℹ️  Both systems use the same base risk scores:")
    print(f"    • billing_complaint: +15.0")
    print(f"    • competitor_mention: +30.0") 
    print(f"    • service_frustration: +20.0")
    print(f"    • process_frustration: +20.0")
    print(f"    • positive_resolution: -40.0")

if __name__ == "__main__":
    test_llm_churn_integration() 