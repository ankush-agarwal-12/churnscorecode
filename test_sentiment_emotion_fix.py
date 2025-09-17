#!/usr/bin/env python3

from simple_churn_scorer import SimpleChurnScorer

def test_sentiment_emotion_extraction():
    """Test that LLM sentiment and emotion are properly extracted"""
    
    print("🔍 TESTING SENTIMENT & EMOTION EXTRACTION")
    print("=" * 60)
    
    # Initialize scorer with LLM enabled
    scorer = SimpleChurnScorer()
    scorer.use_llm_indicators = True
    
    # Test message that should trigger clear sentiment and emotion
    test_message = "I'm extremely frustrated with this service! This is completely unacceptable!"
    agent_context = "I understand your frustration. Let me help resolve this issue."
    
    print(f"Test Message: {test_message}")
    print(f"Agent Context: {agent_context}")
    print("-" * 60)
    
    # Process the message
    event = scorer.process_customer_message(test_message, agent_context)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Sentiment Score: {event.sentiment_score:.3f}")
    print(f"Emotion: {event.emotion_result['dominant_emotion']} (confidence: {event.emotion_result['dominant_score']:.3f})")
    print(f"Risk Patterns: {event.detected_patterns}")
    print(f"Total Risk Delta: {event.risk_delta:.1f}")
    print(f"Churn Score: {scorer.get_current_score()}/100")
    
    print(f"\n✅ Test Complete - Check if sentiment and emotion are from LLM JSON")

if __name__ == "__main__":
    test_sentiment_emotion_extraction() 