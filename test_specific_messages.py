#!/usr/bin/env python3

from llm_indicator_extractor import LLMIndicatorExtractor

def test_specific_messages():
    """Test the specific message combination provided by user"""
    extractor = LLMIndicatorExtractor()
    
    print("🔍 TESTING SPECIFIC MESSAGE COMBINATION")
    print("=" * 70)
    print(f"Customer: {extractor.customer_profile['name']}")
    print(f"Current MRC: ${extractor.customer_profile['current_mrc']}")
    print("=" * 70)
    
    # Build the conversation context from the messages
    conversation_context = """Agent: I understand. Since you mentioned you barely watch TV, Do you want to remove that?
Customer: Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper.
Agent: So your new bill would be around $150/month & I can also add a 10 dollar discount if you subscribe for ebill which would make it $140 going forward.
Customer: How much would that cost me?
Agent: I've updated your plan and removed the rental. You'll see the changes in your next cycle. Anything else I can help you with today?"""
    
    # The current customer message to analyze
    current_message = "Alright, let's go with that then. Please make sure the new charges reflect next month."
    
    print(f"\n📝 CONVERSATION CONTEXT:")
    print(conversation_context)
    
    print(f"\n🎯 CURRENT MESSAGE TO ANALYZE:")
    print(f'"{current_message}"')
    
    print(f"\n" + "-" * 70)
    
    # Get analysis
    analysis = extractor.get_comprehensive_analysis(current_message, conversation_context)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    # Display results
    print(f"\n🤖 ANALYSIS RESULTS:")
    print(f"   📊 SENTIMENT: {analysis['sentiment']}")
    print(f"   😊 EMOTION: {analysis['emotion']}")
    print(f"   🚨 RISK PATTERNS: {analysis['risk_patterns']}")
    
    # Service Analysis
    tv_analysis = analysis['tv_usage_analysis']
    print(f"\n   📺 SERVICE ANALYSIS:")
    print(f"      TV Usage: {tv_analysis['tv_usage_level']}")
    print(f"      TV Removal Interest: {tv_analysis['removal_interest']}")
    
    # Price Analysis
    price_analysis = analysis['price_analysis']
    print(f"\n   💰 PRICE ANALYSIS:")
    print(f"      Price Sensitive: {price_analysis['price_sensitive']}")
    print(f"      Budget Concern: {price_analysis['budget_concern_level']}")
    
    # Show filtered offers
    print(f"\n🎯 FILTERED OFFERS ({len(analysis['filtered_offers'])} total):")
    
    for j, offer in enumerate(analysis['filtered_offers'], 1):
        print(f"\n   {j}. {offer['title']}")
        print(f"      Price: ${offer['price_delta']}/month | Products: {', '.join(offer['product_types'])}")
        
        if 'filtering_explanation' in offer:
            print(f"      Filtering Logic:")
            for explanation in offer['filtering_explanation']:
                print(f"        • {explanation}")
        
        if 'low_usage_reason' in offer:
            print(f"      ⚠️ Low Usage: {offer['low_usage_reason']}")
    
    # Show removed offers if any
    removed_offers = [offer for offer in extractor.initial_offers 
                     if offer['offer_id'] not in [f['offer_id'] for f in analysis['filtered_offers']]]
    
    if removed_offers:
        print(f"\n❌ REMOVED OFFERS ({len(removed_offers)}):")
        for j, offer in enumerate(removed_offers, 1):
            print(f"   {j}. {offer['title']} - Price: ${offer['price_delta']}/month")
            print(f"      Products: {', '.join(offer['product_types'])}")

if __name__ == "__main__":
    test_specific_messages() 