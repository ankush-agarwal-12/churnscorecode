#!/usr/bin/env python3

from llm_indicator_extractor import LLMIndicatorExtractor

def test_full_conversation():
    """Test the entire conversation with rolling window analysis"""
    extractor = LLMIndicatorExtractor()
    
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
    
    print("🔍 FULL CONVERSATION ANALYSIS - ROLLING WINDOW")
    print("=" * 80)
    print(f"Customer: {extractor.customer_profile['name']}")
    print(f"Current MRC: ${extractor.customer_profile['current_mrc']}")
    print(f"Total conversation turns: {len(conversation)}")
    print("=" * 80)
    
    # Find analysis points - every customer message after we have at least 4 messages
    analysis_points = []
    for i in range(len(conversation)):
        speaker, message = conversation[i]
        if speaker == "Customer" and i >= 3:  # Need at least 4 messages for context
            analysis_points.append(i)
    
    print(f"Analysis points (customer messages): {len(analysis_points)}")
    print("=" * 80)
    
    for point_idx, conv_idx in enumerate(analysis_points, 1):
        speaker, customer_message = conversation[conv_idx]
        
        # Get 4 message window: look back to get agent+customer+agent+customer
        # We want the pattern: [agent, customer, agent, customer_current]
        window_start = max(0, conv_idx - 3)  # Get 3 previous + current = 4 total
        window_end = conv_idx + 1
        
        context_messages = conversation[window_start:conv_idx]  # All except current
        context_text = []
        
        for ctx_speaker, ctx_message in context_messages:
            context_text.append(f"{ctx_speaker}: {ctx_message}")
        
        conversation_context = "\n".join(context_text)
        
        print(f"\n📍 ANALYSIS POINT {point_idx} (Turn {conv_idx + 1})")
        print("-" * 50)
        print(f"🗣️ CONVERSATION WINDOW ({len(context_messages)} previous messages):")
        for ctx_speaker, ctx_message in context_messages:
            print(f"   {ctx_speaker}: {ctx_message}")
        
        print(f"\n🎯 CURRENT CUSTOMER MESSAGE:")
        print(f'   "{customer_message}"')
        
        # Get analysis
        analysis = extractor.get_comprehensive_analysis(customer_message, conversation_context)
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            continue
        
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
        
        # Show filtered offers count and key changes
        filtered_count = len(analysis['filtered_offers'])
        total_offers = len(extractor.initial_offers)
        removed_count = total_offers - filtered_count
        
        print(f"\n🎯 OFFER FILTERING IMPACT:")
        print(f"   📋 Total Available: {total_offers}")
        print(f"   ✅ Filtered Offers: {filtered_count}")
        print(f"   ❌ Removed Offers: {removed_count}")
        
        # Show top 3 offers
        if analysis['filtered_offers']:
            print(f"\n   🏆 TOP 3 OFFERS:")
            for j, offer in enumerate(analysis['filtered_offers'][:3], 1):
                print(f"      {j}. {offer['title']} - ${offer['price_delta']}/month")
                print(f"         Products: {', '.join(offer['product_types'])}")
        
        # Show removed offers by category
        removed_offers = [offer for offer in extractor.initial_offers 
                         if offer['offer_id'] not in [f['offer_id'] for f in analysis['filtered_offers']]]
        
        if removed_offers:
            # Group by reason
            tv_removed = [o for o in removed_offers if "TV" in o['product_types']]
            price_removed = [o for o in removed_offers if "TV" not in o['product_types']]
            
            if tv_removed:
                print(f"   🚫 TV Offers Removed: {len(tv_removed)}")
            if price_removed:
                print(f"   💸 Price-Filtered Offers: {len(price_removed)}")
        
        print(f"\n" + "="*80)

if __name__ == "__main__":
    test_full_conversation() 