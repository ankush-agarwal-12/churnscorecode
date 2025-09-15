#!/usr/bin/env python3

import sys
import os
from llm_indicator_extractor import LLMIndicatorExtractor

def test_single_extraction():
    """Test a single indicator extraction"""
    extractor = LLMIndicatorExtractor()
    
    # Sample from your conversation
    customer_msg = "I just opened my bill and it's $200 again. Last month it was $180. Why is it going up every time?"
    agent_msg = "I understand that's frustrating. Let me walk through your bill to identify the changes."
    
    print("Testing LLM Indicator Extraction...")
    print(f"Customer: {customer_msg}")
    print(f"Agent: {agent_msg}")
    print("\nExtracting indicators...")
    
    indicators = extractor.extract_indicators(customer_msg, agent_msg)
    
    if indicators:
        print("\n✅ SUCCESS - Indicators extracted!")
        
        # Show detected risk patterns
        detected_risks = extractor.get_detected_risk_patterns(indicators)
        print(f"\n🚨 Risk Patterns: {detected_risks}")
        
        # Show offer filtering indicators
        offer_context = extractor.get_offer_filtering_context(indicators)
        
        print(f"\n💰 Price Sensitivity:")
        print(f"   - Detected: {offer_context['price_sensitivity']['detected']}")
        print(f"   - Confidence: {offer_context['price_sensitivity']['confidence']:.2f}")
        print(f"   - Mentioned Prices: {offer_context['price_sensitivity']['mentioned_prices']}")
        
        print(f"\n📺 Service Usage Patterns:")
        print(f"   - TV Usage: {offer_context['service_preferences']['tv_usage']}")
        print(f"   - Mobile Usage: {offer_context['service_preferences']['mobile_usage']}")
        print(f"   - Internet Usage: {offer_context['service_preferences']['internet_usage']}")
        print(f"   - Confidence: {offer_context['service_preferences']['confidence']:.2f}")
        
        print(f"\n🎯 Value Preferences:")
        print(f"   - Price Focused: {offer_context['value_drivers']['price_focused']}")
        print(f"   - Feature Focused: {offer_context['value_drivers']['feature_focused']}")
        print(f"   - Convenience Focused: {offer_context['value_drivers']['convenience_focused']}")
        
        print(f"\n⚡ Urgency Indicators:")
        print(f"   - Immediate Resolution Needed: {offer_context['urgency']['immediate_resolution_needed']}")
        print(f"   - Considering Alternatives: {offer_context['urgency']['considering_alternatives']}")
        
        print(f"\n📋 Overall Confidence Score: {indicators.confidence_score:.2f}")
        
    else:
        print("\n❌ FAILED - Could not extract indicators")
        print("Check HuggingFace API connection and token")
        print("Model used: meta-llama/Meta-Llama-3-8B-Instruct")

def test_tv_usage_case():
    """Test the specific TV usage case you mentioned"""
    extractor = LLMIndicatorExtractor()
    
    customer_msg = "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for."
    agent_msg = "I see here that your promotional discount expired a month ago, and there's a rental charge for a second set-top box."
    
    print("\n" + "="*60)
    print("Testing TV Usage Case...")
    print(f"Customer: {customer_msg}")
    print(f"Agent: {agent_msg}")
    
    indicators = extractor.extract_indicators(customer_msg, agent_msg)
    
    if indicators:
        offer_context = extractor.get_offer_filtering_context(indicators)
        
        print(f"\n📺 TV Usage Analysis:")
        print(f"   - TV Usage Level: {offer_context['service_preferences']['tv_usage']}")
        print(f"   - Confidence: {offer_context['service_preferences']['confidence']:.2f}")
        
        print(f"\n💡 Key Point: Even if TV usage is 'low' or 'unused', this doesn't mean")
        print(f"   we automatically remove ALL TV offers. Instead, we can:")
        print(f"   - Prioritize cheaper TV packages")
        print(f"   - Focus on bundle discounts")
        print(f"   - Highlight value propositions")
        print(f"   - Consider TV-free alternatives")
        
    else:
        print("❌ Failed to extract TV usage indicators")

if __name__ == "__main__":
    print("🧪 LLM Indicator Extraction Test")
    print("="*60)
    
    # Test basic extraction
    test_single_extraction()
    
    # Test TV usage case
    test_tv_usage_case()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("\nNext steps:")
    print("1. Refine prompts based on results")
    print("2. Adjust confidence thresholds")
    print("3. Integrate with churn scorer logic") 