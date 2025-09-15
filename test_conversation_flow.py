#!/usr/bin/env python3

from llm_indicator_extractor import LLMIndicatorExtractor
from typing import List, Tuple

def analyze_conversation_flow():
    """Analyze the full conversation from simple_churn_scorer.py using rolling windows"""
    
    # Full conversation from simple_churn_scorer.py
    conversation = [
        ("Agent", "Thank you for calling customer Service, this is Jason speaking. May I have your account number please?"),
        ("Customer", "Sure, it's 29871003."),
        ("Agent", "Thanks, Mark. I'll need to verify your identity. I've just sent a 4-digit verification code to your registered mobile number ending in 6024. Could you read that out for me?"),
        ("Customer", "Yep, it's 9384."),
        ("Agent", "Perfect. Give me a moment while I pull up your account details... this may take 30 seconds. Do you mind holding?"),
        ("Customer", "That's fine."),
        ("Agent", "Thanks for waiting. I've got your account up. How can I help you today?"),
        ("Customer", "I just opened my bill and it's $200 again. Last month it was $180. Why is it going up every time?"),
        ("Agent", "I understand that's frustrating. Let me walk through your bill to identify the changes."),
        ("Customer", "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for."),
        ("Agent", "I see here that your promotional discount expired a month ago, and there's a rental charge for a second set-top box."),
        ("Customer", "No one told me the promo would end. Why wouldn't you notify me? This is not okay."),
        ("Agent", "You're absolutely right, and I apologize. We should have communicated that better. We might be able to lower your bill and give you better speeds or extras where it actually matters. Just a sec while I check what we can offer. Would you mind holding again for 30 seconds?"),
        ("Customer", "Sure."),
        ("Agent", "Thanks for holding. I've checked and we can offer you a plan that provides faster internet while keeping your TV, mobile and cybersecurity products at 185 dollar a month"),
        ("Customer", "Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper."),
        ("Agent", "I understand. Since you mentioned you barely watch TV, Do you want to remove that?"),
        ("Customer", "How much would that cost me?"),
        ("Agent", "So your new bill would be around $150/month & I can also add a 10 dollar discount if you subscribe for ebill which would make it $140 going forward. Your total savings would be 60 dollars a month"),
        ("Customer", "Okay, that sounds better. I appreciate you trying. But still, it feels like I have to call every few months just to keep the price reasonable. Just to confirm, I would still be keeping all the mobile lines right?"),
        ("Agent", "That's fair feedback, Sarah. I'll note that on your account. You shouldn't have to go through that and yes, you would keep all your products except TV"),
        ("Customer", "Alright, let's go with that then. Please make sure the new charges reflect next month."),
        ("Agent", "I've updated your plan and removed the rental. You'll see the changes in your next cycle. Anything else I can help you with today?"),
        ("Customer", "No, that's all. Thanks for the help, Jason."),
        ("Agent", "You're welcome. Thanks for being with HorizonConnect. Have a great day!")
    ]
    
    extractor = LLMIndicatorExtractor()
    
    # Define key analysis points in the conversation
    analysis_points = [
        {
            "window_start": 6,  # Start from when real conversation begins
            "window_end": 10,   # First billing complaint
            "description": "Initial Billing Complaint - Customer reveals bill went from $180 to $200"
        },
        {
            "window_start": 8,
            "window_end": 12,
            "description": "TV Usage Revelation - Customer says 'I barely watch TV'"
        },
        {
            "window_start": 10,
            "window_end": 14,
            "description": "Process Frustration - Customer upset about promo notification"
        },
        {
            "window_start": 13,
            "window_end": 17,
            "description": "Competitor Mention - Customer mentions TMobile"
        },
        {
            "window_start": 15,
            "window_end": 19,
            "description": "Price Negotiation - Agent offers $150 then $140"
        },
        {
            "window_start": 18,
            "window_end": 22,
            "description": "Resolution & Process Concern - Customer accepts but mentions recurring issues"
        },
        {
            "window_start": 20,
            "window_end": 24,
            "description": "Final Resolution - Customer accepts TV removal"
        }
    ]
    
    print("🔍 CONVERSATION FLOW ANALYSIS")
    print("=" * 80)
    print("Analyzing how risk patterns and offer indicators evolve throughout the conversation")
    print("Each analysis uses a 4-5 message window to simulate real-time processing")
    print("=" * 80)
    
    for i, analysis_point in enumerate(analysis_points, 1):
        print(f"\n📍 ANALYSIS POINT {i}: {analysis_point['description']}")
        print("-" * 60)
        
        # Extract the conversation window
        window = conversation[analysis_point['window_start']:analysis_point['window_end']]
        
        # Get the final customer message from the window
        final_customer_message = get_final_customer_message(window)
        
        if not final_customer_message:
            print("   ⚠️  No customer message found in this window")
            continue
        
        # Get previous context (all messages before the final customer message)
        context_messages = get_context_messages(window, final_customer_message)
        
        print("🗣️  CONVERSATION WINDOW:")
        for speaker, message in window:
            print(f"   {speaker}: {message}")
        
        print(f"\n🎯 ANALYZING FINAL CUSTOMER MESSAGE:")
        print(f"   Customer: {final_customer_message}")
        
        print(f"\n🤖 LLM ANALYSIS:")
        
        # Extract indicators focusing on final customer message with context
        indicators = extractor.extract_indicators(final_customer_message, context_messages)
        
        if indicators:
            analyze_indicators_simple(indicators)
        else:
            print("   ❌ Failed to extract indicators")
        
        print("\n" + "="*60)
    
    print("\n🎯 KEY INSIGHTS FOR OFFER FILTERING:")
    print("=" * 50)
    print("Based on the conversation flow analysis:")
    print("1. TV offers should be DEPRIORITIZED (not removed) when:")
    print("   - tv_usage detected as 'low' or 'unused'")
    print("   - Customer explicitly mentions barely watching TV")
    print("   - Price sensitivity is high")
    print("")
    print("2. TV offers should be REMOVED when:")
    print("   - Customer explicitly asks to remove TV service")
    print("   - Agent suggests TV removal and customer shows interest")
    print("")
    print("3. Offer filtering scenarios from this conversation:")
    print("   - Point 2: Detect TV usage as 'low' → Show TV-free bundles")
    print("   - Point 4: Competitor mention → Price-focused offers")
    print("   - Point 5: Price negotiation → Budget-friendly options")
    print("   - Point 7: TV removal accepted → Internet + Mobile only")

def format_conversation_window(window: List[Tuple[str, str]]) -> str:
    """Format conversation window for LLM analysis"""
    formatted_lines = []
    for speaker, message in window:
        formatted_lines.append(f"{speaker}: {message}")
    return "\n".join(formatted_lines)

def analyze_indicators(indicators, context_description):
    """Analyze and display the extracted indicators"""
    
    # Risk Pattern Analysis
    detected_risks = []
    risk_attrs = [
        ("billing_complaint", indicators.risk_indicators.billing_complaint),
        ("competitor_mention", indicators.risk_indicators.competitor_mention),
        ("service_frustration", indicators.risk_indicators.service_frustration),
        ("process_frustration", indicators.risk_indicators.process_frustration),
        ("positive_resolution", indicators.risk_indicators.positive_resolution)
    ]
    
    for pattern_name, pattern_data in risk_attrs:
        if pattern_data.get("detected", False):
            evidence = pattern_data.get("evidence", "")
            detected_risks.append(f"{pattern_name}: '{evidence}'")
    
    print(f"   🚨 RISK PATTERNS: {detected_risks if detected_risks else ['None detected']}")
    
    # Offer Indicator Analysis
    offer_context = get_offer_filtering_context(indicators)
    
    print(f"   💰 PRICE SENSITIVITY: {offer_context['price_sensitivity']['detected']}")
    if offer_context['price_sensitivity']['mentioned_prices']:
        print(f"       Mentioned Prices: {offer_context['price_sensitivity']['mentioned_prices']}")
    
    print(f"   📺 SERVICE USAGE:")
    print(f"       TV: {offer_context['service_preferences']['tv_usage']}")
    print(f"       Mobile: {offer_context['service_preferences']['mobile_usage']}")
    print(f"       Internet: {offer_context['service_preferences']['internet_usage']}")
    
    print(f"   🎯 VALUE FOCUS:")
    print(f"       Price Focused: {offer_context['value_drivers']['price_focused']}")
    print(f"       Feature Focused: {offer_context['value_drivers']['feature_focused']}")
    
    print(f"   ⚡ URGENCY: {offer_context['urgency']['urgency_high']}")

def get_final_customer_message(window: List[Tuple[str, str]]) -> str:
    """Extracts the final customer message from a conversation window."""
    # Go through the window in reverse to find the last customer message
    for speaker, message in reversed(window):
        if speaker == "Customer":
            return message
    return ""

def get_context_messages(window: List[Tuple[str, str]], final_customer_message: str) -> str:
    """Extracts all messages before the final customer message as context."""
    context_messages = []
    found_final = False
    
    # Go through the window in reverse to find where the final customer message is
    for i, (speaker, message) in enumerate(window):
        if speaker == "Customer" and message == final_customer_message:
            # Get all messages before this one as context
            for j in range(i):
                prev_speaker, prev_message = window[j]
                context_messages.append(f"{prev_speaker}: {prev_message}")
            break
    
    return "\n".join(context_messages)

def analyze_indicators_simple(indicators):
    """Analyze and display the extracted indicators in a simplified format."""
    
    # Risk Pattern Analysis
    detected_risks = []
    risk_attrs = [
        ("billing_complaint", indicators.risk_indicators.billing_complaint),
        ("competitor_mention", indicators.risk_indicators.competitor_mention),
        ("service_frustration", indicators.risk_indicators.service_frustration),
        ("process_frustration", indicators.risk_indicators.process_frustration),
        ("positive_resolution", indicators.risk_indicators.positive_resolution)
    ]
    
    for pattern_name, pattern_data in risk_attrs:
        if pattern_data.get("detected", False):
            evidence = pattern_data.get("evidence", "")
            detected_risks.append(f"{pattern_name}: '{evidence}'")
    
    print(f"   🚨 RISK PATTERNS: {detected_risks if detected_risks else ['None detected']}")
    
    # Offer Indicator Analysis
    offer_context = get_offer_filtering_context(indicators)
    
    print(f"   💰 PRICE SENSITIVITY: {offer_context['price_sensitivity']['detected']}")
    if offer_context['price_sensitivity']['mentioned_prices']:
        print(f"       Mentioned Prices: {offer_context['price_sensitivity']['mentioned_prices']}")
    
    print(f"   📺 SERVICE USAGE:")
    print(f"       TV: {offer_context['service_preferences']['tv_usage']}")
    print(f"       Mobile: {offer_context['service_preferences']['mobile_usage']}")
    print(f"       Internet: {offer_context['service_preferences']['internet_usage']}")
    
    print(f"   🎯 VALUE FOCUS:")
    print(f"       Price Focused: {offer_context['value_drivers']['price_focused']}")
    print(f"       Feature Focused: {offer_context['value_drivers']['feature_focused']}")
    
    print(f"   ⚡ URGENCY: {offer_context['urgency']['urgency_high']}")

def get_offer_filtering_context(indicators):
    """Extract offer filtering context (simplified version)"""
    return {
        "price_sensitivity": {
            "detected": indicators.offer_indicators.price_sensitivity.get("detected", False),
            "mentioned_prices": indicators.offer_indicators.price_sensitivity.get("mentioned_prices", [])
        },
        "service_preferences": {
            "tv_usage": indicators.offer_indicators.service_usage_patterns.get("tv_usage", "unknown"),
            "mobile_usage": indicators.offer_indicators.service_usage_patterns.get("mobile_usage", "unknown"),
            "internet_usage": indicators.offer_indicators.service_usage_patterns.get("internet_usage", "unknown")
        },
        "value_drivers": {
            "price_focused": indicators.offer_indicators.value_preferences.get("price_focused", False),
            "feature_focused": indicators.offer_indicators.value_preferences.get("feature_focused", False)
        },
        "urgency": {
            "urgency_high": indicators.offer_indicators.urgency_indicators.get("urgency_high", False)
        }
    }

if __name__ == "__main__":
    print("🧪 CONVERSATION FLOW ANALYSIS TEST")
    print("Testing LLM indicator extraction across the full conversation timeline")
    print("This simulates how indicators would be detected every few seconds/messages")
    print("")
    
    analyze_conversation_flow() 