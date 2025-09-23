# #!/usr/bin/env python3

# import os
# from llm_indicator_extractor import LLMIndicatorExtractor

# def test_llm_conversation():
#     """
#     Test script for LLM-only mode: Processes a conversation using LLMIndicatorExtractor
#     to extract indicators, analyze sentiment/emotion, and filter offers based on the conversation.
#     Demonstrates multi-step offer filtering without WebSocket or other integrations.
#     """
    
#     # Ensure HuggingFace token is set
#     if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
#         print("❌ Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
#         return
    
#     # Initialize LLM Indicator Extractor
#     extractor = LLMIndicatorExtractor()
    
#     # Sample conversation to test
#     conversation1 = [
#         ("Agent", "Thanks for calling HorizonConnect, this is Emily. May I have your account number?"),
#         ("Customer", "Sure, it’s 50982317."),
#         ("Agent", "Thanks, Mr. Khan. How can I help today?"),
#         ("Customer", "My contract ends next month. I don’t want to renew for another year. I need something short-term."),
#         ("Agent", "I understand. I’m seeing here we can renew you with a 2-year plan at $120/month, plus free streaming subscription."),
#         ("Customer", "Two years? No, I move a lot for work. I don’t want to be tied down."),
#         ("Agent", "Okay, the shortest promotion we currently offer is a 1-year plan at $125/month."),
#         ("Customer", "That’s still too long. I only want month-to-month."),
#         ("Agent", "Month-to-month is possible, but it would be $160/month without discounts."),
#         ("Customer", "That’s not worth it. Basically, you’re forcing me into a contract."),
#         ("Agent", "I understand it feels restrictive, but that’s how our offers are structured."),
#         ("Customer", "Then I’ll have to cancel and look at competitors. Sorry."),
#         ("Agent", "I respect your decision. Let me process the cancellation.")
#     ]
    
#     print("🤖 LLM-ONLY CONVERSATION TEST")
#     print("=" * 80)
#     print(f"Total conversation turns: {len(conversation1)}")
#     print("Mode: LLM-based indicator extraction and offer filtering")
#     print("=" * 80)
    
#     last_agent_message = ""
#     customer_count = 0
    
#     # Process each turn in the conversation
#     for i, (speaker, text) in enumerate(conversation1):
#         if speaker == "Customer":
#             customer_count += 1
#             print(f"\n📍 Turn {i+1} - Customer Message #{customer_count}")
#             print(f"💬 Customer: \"{text}\"")
#             print("-" * 60)
            
#             # Get comprehensive LLM analysis
#             analysis = extractor.get_comprehensive_analysis(text, last_agent_message)
            
#             if "error" in analysis:
#                 print(f"❌ LLM Analysis Failed: {analysis['error']}")
#                 continue
            
#             # Display LLM Analysis
#             print(f"🤖 LLM ANALYSIS:")
#             print(f"   📊 Sentiment: {analysis['sentiment']}")
#             print(f"   😊 Emotion: {analysis['emotion']}")
#             print(f"   🚨 Risk Patterns: {analysis['risk_patterns']}")
            
#             # Show detailed indicators
#             offer_indicators = analysis.get('offer_indicators', {})
#             tv_analysis = analysis.get('tv_usage_analysis', {})
#             price_analysis = analysis.get('price_analysis', {})
            
#             print(f"   📺 TV Usage Analysis: {tv_analysis.get('tv_usage_level', 'unknown')} | Removal Interest: {tv_analysis.get('removal_interest', False)}")
#             print(f"   💰 Price Analysis: Budget Concern: {price_analysis.get('budget_concern_level', 'none')} | Sensitive: {price_analysis.get('price_sensitive', False)}")
#             print(f"   📋 Service Usage: {offer_indicators.get('service_usage', {})}")
#             print(f"   🔧 Value Preference: {offer_indicators.get('value_preference', 'balanced')}")
            
#             # Get and display filtered offers
#             filtered_offers = analysis['filtered_offers']
#             print(f"\n🎯 FILTERED OFFERS ({len(filtered_offers)} total):")
            
#             if filtered_offers:
#                 for j, offer in enumerate(filtered_offers, 1):
#                     print(f"   {j}. {offer['title']}")
#                     print(f"      Price: ${offer['price_delta']}/month | Products: {', '.join(offer['product_types'])}")
#                     print(f"      Contract: {offer['contract_type']} | Retention: {offer['retention_offer']}")
                    
#                     # Show filtering explanations to demonstrate multi-step filtering
#                     if 'filtering_explanation' in offer:
#                         print(f"      📝 Filtering Logic:")
#                         for explanation in offer['filtering_explanation']:
#                             print(f"         • {explanation}")
                    
#                     if 'acceptance_reason' in offer:
#                         print(f"      ✅ Acceptance Reason: {offer['acceptance_reason']}")
                    
#                     print()
            
#             # Show removed offers to highlight filtering
#             all_offers = extractor.initial_offers
#             removed_offers = [offer for offer in all_offers if offer['offer_id'] not in [f['offer_id'] for f in filtered_offers]]
#             if removed_offers:
#                 print(f"   ❌ REMOVED OFFERS ({len(removed_offers)}):")
#                 for j, offer in enumerate(removed_offers[:5], 1):  # Show first 5 for brevity
#                     print(f"      {j}. {offer['title']} - Price: ${offer['price_delta']}/month | Reason: {offer.get('removal_reason', 'Filtered out by budget/usage')}")
#                 if len(removed_offers) > 5:
#                     print(f"      ... and {len(removed_offers) - 5} more")
            
#             print("=" * 80)
        
#         else:  # Agent
#             last_agent_message = text

# if __name__ == "__main__":
#     test_llm_conversation()


#!/usr/bin/env python3

import os
from llm_indicator_extractor import LLMIndicatorExtractor
from simple_churn_scorer import SimpleChurnScorer  # For churn score calculation

def test_llm_conversation():
    """
    Test script for LLM-only mode: Processes conversations using LLMIndicatorExtractor
    and SimpleChurnScorer to extract indicators, analyze sentiment/emotion, filter offers,
    and calculate churn risk scores. Demonstrates multi-step offer filtering.
    """
    
    # Ensure HuggingFace token is set
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("❌ Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
        return
    
    # Initialize LLM Indicator Extractor
    extractor = LLMIndicatorExtractor()
    
    # Sample conversations to test
    conversations = {
        "Conversation 1": [
    # ("Agent", "Thank you for calling us, this is Alina. Can I have your account number please?"),
    # ("Customer", "Yeah, it's 75299134."),
    # ("Agent", "Thanks. And just to verify the account, am I speaking with Stephanie?"),
    # ("Customer", "Yes, that's me."),
    # ("Agent", "Great. For your security, I've just sent a 4-digit code to your mobile ending in 6641. Could you read it back for me?"),
    # ("Customer", "One sec <pause 1 sec> okay, got it. It's 4472."),
    # ("Agent", "Perfect, you're verified. Let me bring up your account details. This might take just a few seconds... Do you mind holding?"),
    # ("Customer", "Sure."),
    # ("Agent", "Thanks for waiting, Stephanie. I have your account open. How can I help you today?"),
    # ("Customer", "I'm calling to cancel my service."),
    # ("Agent", "Oh, I'm certainly sorry to hear that. We'd hate to lose you. May I ask what's led you to this decision?"),
    # ("Customer", "I just got an offer from Horizon for 1 Gbps internet plus their basic TV for $85 a month. That's $50 less than what I'm paying you."),
    # ("Agent", "I see. $85 for a Gig plan is a very aggressive offer. I'm looking at your current bill, and I see you're at $135 for our 400 Mbps plan and the 'Platinum' TV tier. I can definitely see the price gap."),
    # ("Customer", "Exactly. And frankly, I'm tired of it. I've been paying $135 forever, and every single year I have to call in and beg for some new discount. It's exhausting."),
    # ("Agent", "That makes perfect sense, and I truly apologize for that experience. It should not feel that way, especially for a loyal customer like yourself. I see you've been with us for over 8 years. Let me pull up our current retention offers to see if we can find something that better fits your needs and budget. One moment."),
    ("Agent", "Okay, thanks for holding. Because of your tenure, I can offer you our 1 Gbps + Base TV package for $95 per month — but this would require a 24-month agreement."),
    ("Customer", "There’s the catch. I don’t want to be locked in for 2 years. What if I need to move or switch again?"),
    ("Agent", "I understand. In that case, the shortest commitment I can apply is 12 months, at $100 per month."),
    ("Customer", "Even 12 months feels too long. I want the flexibility to go month-to-month."),
    ("Agent", "We do have a true month-to-month option. It’s $105 per month for the same 1 Gbps + Base TV plan. No contract, cancel anytime."),
    ("Customer", "So $105 monthly with no strings attached?"),
    ("Agent", "Exactly. You’ll pay a little more compared to the contract rates, but you get full flexibility and can cancel or change anytime without penalties."),
    ("Customer", "That’s the peace of mind I want. Okay, let’s go with the $105 monthly, no contract."),
    ("Agent", "Perfect. I’ll update your account right now. Your new month-to-month plan will begin with your next billing cycle, and you’ll see the new rate reflected immediately. Is there anything else I can assist you with today?"),
    ("Customer", "No, that’s it. Thanks for working with me, Alina."),
    ("Agent", "You’re very welcome. Thank you for being with us, and enjoy your upgraded plan with full flexibility. Have a wonderful day.")
]
        # "Conversation 2": [
        #     ("Agent", "Hello, you’re speaking with Raj from HorizonConnect. Can I have your account number?"),
        #     ("Customer", "It’s 77459120."),
        #     ("Agent", "Thank you, Ms. Lopez. What’s the issue today?"),
        #     ("Customer", "My video calls keep freezing, especially during work hours. I need reliable 5G speed."),
        #     ("Agent", "Sorry about that. I see you’re on our standard 5G plan. We can upgrade you for $10 less if you take our 18-month contract."),
        #     ("Customer", "I don’t care about $10 savings. I want consistent speeds and priority network access."),
        #     ("Agent", "Okay, we have a ‘Premium Unlimited’ plan with priority bandwidth. But it requires a 2-year lock-in."),
        #     ("Customer", "I don’t want long contracts. Is there a monthly option?"),
        #     ("Agent", "Unfortunately, premium priority is only bundled with long-term commitments."),
        #     ("Customer", "Then this doesn’t help. I’d rather pay more for short-term with better features."),
        #     ("Agent", "At the moment, that’s not available. I completely understand if you need to explore other providers."),
        #     ("Customer", "Yeah, I think I will. Thanks.")
        # ],
        # "Conversation 3": [
        #     ("Agent", "Hi, this is Jason from HorizonConnect. How may I assist?"),
        #     ("Customer", "My bill shot up to $210 this month. Can you explain why?"),
        #     ("Agent", "I see two pay-per-view charges and your introductory discount ended."),
        #     ("Customer", "I didn’t order those pay-per-views. Must be a mistake."),
        #     ("Agent", "I understand. I can’t remove them since they were recorded by your box. What I can do is move you to a bundle plan at $195/month for 24 months."),
        #     ("Customer", "That’s still more than before, and I don’t want a 24-month contract."),
        #     ("Agent", "The shorter option is $200/month for 12 months, but fewer channels."),
        #     ("Customer", "That’s not a deal. I just want my old $170 back."),
        #     ("Agent", "I wish I could restore that, but it’s not available anymore."),
        #     ("Customer", "Then I’m going to cancel after this cycle."),
        #     ("Agent", "I understand. Let me note that down.")
        # ]
    }
    
    for conv_name, conversation in conversations.items():
        print(f"\n{'='*80}")
        print(f"🤖 TESTING: {conv_name}")
        print(f"{'='*80}")
        print(f"Total turns: {len(conversation)} | MRC: $140")
        
        # Initialize scorer for churn risk calculation
        scorer = SimpleChurnScorer()
        scorer.use_llm_indicators = True
        scorer.use_llm_offer_filtering = True
        if scorer.llm_extractor is None:
            scorer.llm_extractor = extractor  # Reuse extractor
        
        last_agent_message = ""
        customer_count = 0
        cumulative_churn_score = 50.0  # Starting score
        
        for i, (speaker, text) in enumerate(conversation):
            if speaker == "Customer":
                customer_count += 1
                print(f"\n📍 Turn {i+1} - Customer #{customer_count}: {text[:50]}...")
                
                # Calculate churn risk
                event = scorer.process_customer_message(text, last_agent_message)
                cumulative_churn_score = event.cumulative_score
                print(f"   🚨 Churn Score: {cumulative_churn_score:.1f}/100 (Delta: {event.risk_delta:.1f})")
                
                # LLM Analysis and Offer Filtering
                analysis = scorer.llm_extractor.get_comprehensive_analysis(text, last_agent_message)
                if "error" in analysis:
                    print(f"   ❌ LLM Failed: {analysis['error']}")
                    continue
                
                print(f"   📊 Sentiment: {analysis['sentiment']} | Emotion: {analysis['emotion']}")
                print(f"   🚨 Risk Patterns: {analysis['risk_patterns']}")
                
                # Show filtered offers
                filtered_offers = analysis['filtered_offers']
                print(f"   🎯 Filtered Offers ({len(filtered_offers)}):")
                for j, offer in enumerate(filtered_offers[:3], 1):
                    print(f"      {j}. {offer['title']} - ${offer['price_delta']}/mo | Filtering: {offer['filtering_explanation'][-1]}")
            
            else:
                last_agent_message = text
        
        print(f"\n🏁 FINAL FOR {conv_name}: Churn Score: {cumulative_churn_score:.1f}/100")

if __name__ == "__main__":
    test_llm_conversation()