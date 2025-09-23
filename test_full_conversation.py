# #!/usr/bin/env python3

# from simple_churn_scorer import SimpleChurnScorer

# def test_llm_churn_and_offers():
#     """Test LLM-based churn scoring and offer filtering with full conversation"""
    
#     # Full conversation from simple_churn_scorer.py
#     conversation =  [
#     ("Agent", "Thanks for calling HorizonConnect, this is Emily. May I have your account number?"),
#     ("Customer", "Sure, it’s 50982317."),
#     ("Agent", "Thanks, Mr. Khan. How can I help today?"),
#     ("Customer", "My contract ends next month. I don’t want to renew for another year. I need something short-term."),
#     ("Agent", "I understand. I’m seeing here we can renew you with a 2-year plan at $120/month, plus free streaming subscription."),
#     ("Customer", "Two years? No, I move a lot for work. I don’t want to be tied down."),
#     ("Agent", "Okay, the shortest promotion we currently offer is a 1-year plan at $125/month."),
#     ("Customer", "That’s still too long. I only want month-to-month."),
#     ("Agent", "Month-to-month is possible, but it would be $160/month without discounts."),
#     ("Customer", "That’s not worth it. Basically, you’re forcing me into a contract."),
#     ("Agent", "I understand it feels restrictive, but that’s how our offers are structured."),
#     ("Customer", "Then I’ll have to cancel and look at competitors. Sorry."),
#     ("Agent", "I respect your decision. Let me process the cancellation.")
# ]


    
#     print("🤖 LLM CHURN & OFFER ANALYSIS")
#     print("=" * 80)
#     print(f"Total conversation turns: {len(conversation)}")
#     print("Mode: LLM-based churn scoring + LLM-based offer filtering")
#     print("=" * 80)
    
#     # Create LLM-enabled scorer
#     scorer = SimpleChurnScorer()
#     scorer.use_llm_indicators = True  # LLM churn scoring
#     scorer.use_llm_offer_filtering = True  # LLM offer filtering
    
#     print(f"🔧 Configuration:")
#     print(f"   LLM Churn Scoring: {scorer.use_llm_indicators}")
#     print(f"   LLM Offer Filtering: {scorer.use_llm_offer_filtering}")
#     print(f"   Starting Score: {scorer.current_score}")
#     print()
    
#     last_agent_message = ""
#     customer_count = 0
    
#     # Process conversation
#     for i, (speaker, text) in enumerate(conversation):
#         if speaker == "Customer":
#             customer_count += 1
#             print(f"\n📍 Turn {i+1} - Customer Message #{customer_count}")
#             print(f"💬 Customer: \"{text}\"")
#             print("-" * 60)
            
#             # Process with LLM
#             event = scorer.process_customer_message(text, last_agent_message)
            
#             print(f"🤖 LLM CHURN ANALYSIS:")
#             print(f"   Sentiment: {event.emotion_result.get('sentiment', 'N/A')} (score: {event.sentiment_score:.3f})")
#             print(f"   Emotion: {event.emotion_result['dominant_emotion']}")
#             print(f"   Risk Patterns: {event.detected_patterns}")
#             print(f"   Risk Delta: {event.risk_delta:.1f}")
#             print(f"   Churn Score: {event.cumulative_score:.1f}/100")
            
#             # Get LLM offers for key customer messages
#             if customer_count >= 2:  # Start showing offers from 2nd customer message
#                 print(f"\n🎯 LLM OFFER FILTERING:")
                
#                 # Get LLM analysis for offers
#                 llm_analysis = scorer.llm_extractor.get_comprehensive_analysis(text, last_agent_message)
                
#                 if "error" not in llm_analysis:
#                     offers = scorer.get_offers_for_agent_with_analysis(text, llm_analysis)
                    
#                     # Show LLM analysis details
#                     print(f"   📊 LLM Analysis:")
#                     print(f"      Sentiment: {llm_analysis.get('sentiment', 'unknown')}")
#                     print(f"      Emotion: {llm_analysis.get('emotion', 'unknown')}")
#                     print(f"      Risk Patterns: {llm_analysis.get('risk_patterns', [])}")
                    
#                     # Show service indicators
#                     offer_indicators = llm_analysis.get('offer_indicators', {})
#                     service_usage = offer_indicators.get('service_usage', {})
#                     service_removal = offer_indicators.get('service_removal_interest', {})
#                     budget_concern = offer_indicators.get('budget_concern_level', 'none')
                    
#                     print(f"      Service Usage: {service_usage}")
#                     print(f"      Removal Interest: {service_removal}")
#                     print(f"      Budget Concern: {budget_concern}")
                    
#                     # Show offer results
#                     accepted = [o for o in offers if o.get('accepted', True)]
#                     rejected = [o for o in offers if not o.get('accepted', True)]
#                     persistent = [o for o in rejected if o.get('persistent_rejection', False)]
#                     new_rejected = [o for o in rejected if not o.get('persistent_rejection', False)]
                    
#                     print(f"\n   📋 Offer Results:")
#                     print(f"      Total: {len(offers)} | Accepted: {len(accepted)} | Rejected: {len(rejected)} ({len(persistent)} persistent)")
                    
#                     if accepted:
#                         print(f"   ✅ Accepted Offers:")
#                         for j, offer in enumerate(accepted[:3], 1):
#                             price_str = f"${offer['price_delta']:+}" if offer['price_delta'] != 0 else "No cost"
#                             print(f"      {j}. {offer['title']} - {price_str}/month")
                    
#                     if new_rejected:
#                         print(f"   🆕 Newly Rejected:")
#                         for j, offer in enumerate(new_rejected[:2], 1):
#                             reason = offer.get('rejection_reason', 'No reason')[:60]
#                             print(f"      {j}. {offer['title']} - {reason}...")
                    
#                     if persistent:
#                         print(f"   📋 Persistent Rejections: {len(persistent)} offers remain rejected")
                
#                 else:
#                     print(f"   ❌ LLM analysis failed: {llm_analysis['error']}")
            
#             print("=" * 80)
        
#         else:  # Agent
#             last_agent_message = text
    
#     # Final summary
#     print(f"\n🏁 FINAL LLM ANALYSIS SUMMARY")
#     print(f"   Final Churn Score: {scorer.current_score:.1f}/100")
#     print(f"   Score Change: {scorer.current_score - 50.0:+.1f}")
#     print(f"   Customer Messages: {customer_count}")
    
#     # Show conversation context
#     if hasattr(scorer, 'conversation_context'):
#         context = scorer.conversation_context
#         print(f"   💰 Prices Mentioned: {context.mentioned_prices}")
#         print(f"   📺 Unused Services: {context.unused_services}")
#         print(f"   🏃 Competitors: {context.competitor_mentions}")
    
#     # Show persistent rejected offers
#     if hasattr(scorer, 'llm_rejected_offers') and scorer.llm_rejected_offers:
#         print(f"\n🚫 PERSISTENT LLM REJECTIONS ({len(scorer.llm_rejected_offers)} offers):")
#         for offer_id, reason in scorer.llm_rejected_offers.items():
#             print(f"   {offer_id}: {reason[:80]}...")

# if __name__ == "__main__":
#     test_llm_churn_and_offers() 


#!/usr/bin/env python3

from simple_churn_scorer import SimpleChurnScorer
from llm_indicator_extractor import LLMIndicatorExtractor

def test_llm_churn_and_offers():
    """Test LLM-based churn scoring and offer filtering with full conversation"""
    
    # Full conversation from simple_churn_scorer.py
    conversation =  [
    ("Agent", "Thanks for calling HorizonConnect, this is Emily. May I have your account number?"),
    ("Customer", "Sure, it’s 50982317."),
    ("Agent", "Thanks, Mr. Khan. How can I help today?"),
    ("Customer", "My contract ends next month. I don’t want to renew for another year. I need something short-term."),
    ("Agent", "I understand. I’m seeing here we can renew you with a 2-year plan at $120/month, plus free streaming subscription."),
    ("Customer", "Two years? No, I move a lot for work. I don’t want to be tied down."),
    ("Agent", "Okay, the shortest promotion we currently offer is a 1-year plan at $125/month."),
    ("Customer", "That’s still too long. I only want month-to-month."),
    ("Agent", "Month-to-month is possible, but it would be $160/month without discounts."),
    ("Customer", "That’s not worth it. Basically, you’re forcing me into a contract."),
    ("Agent", "I understand it feels restrictive, but that’s how our offers are structured."),
    ("Customer", "Then I’ll have to cancel and look at competitors. Sorry."),
    ("Agent", "I respect your decision. Let me process the cancellation.")
]


    
    print("🤖 LLM CHURN & OFFER ANALYSIS")
    print("=" * 80)
    print(f"Total conversation turns: {len(conversation)}")
    print("Mode: LLM-based churn scoring + LLM-based offer filtering")
    print("=" * 80)
    
    # Create LLM-enabled scorer
    scorer = SimpleChurnScorer()
    scorer.use_llm_indicators = True  # LLM churn scoring
    scorer.use_llm_offer_filtering = True  # LLM offer filtering
    
    # Initialize LLM extractor early if using LLM offer filtering
    if scorer.use_llm_offer_filtering:
        scorer.llm_extractor = LLMIndicatorExtractor()
    
    print(f"🔧 Configuration:")
    print(f"   LLM Churn Scoring: {scorer.use_llm_indicators}")
    print(f"   LLM Offer Filtering: {scorer.use_llm_offer_filtering}")
    print(f"   Starting Score: {scorer.current_score}")
    print()
    
    last_agent_message = ""
    customer_count = 0
    
    # Process conversation
    for i, (speaker, text) in enumerate(conversation):
        if speaker == "Customer":
            customer_count += 1
            print(f"\n📍 Turn {i+1} - Customer Message #{customer_count}")
            print(f"💬 Customer: \"{text}\"")
            print("-" * 60)
            
            # Process with LLM
            event = scorer.process_customer_message(text, last_agent_message)
            
            print(f"🤖 LLM CHURN ANALYSIS:")
            print(f"   Sentiment: {event.emotion_result.get('sentiment', 'N/A')} (score: {event.sentiment_score:.3f})")
            print(f"   Emotion: {event.emotion_result['dominant_emotion']}")
            print(f"   Risk Patterns: {event.detected_patterns}")
            print(f"   Risk Delta: {event.risk_delta:.1f}")
            print(f"   Churn Score: {event.cumulative_score:.1f}/100")
            
            # Get LLM offers for key customer messages
            if customer_count >= 2:  # Start showing offers from 2nd customer message
                print(f"\n🎯 LLM OFFER FILTERING:")
                
                # Get LLM analysis for offers
                llm_analysis = scorer.llm_extractor.get_comprehensive_analysis(text, last_agent_message)
                
                if "error" not in llm_analysis:
                    offers = scorer.get_offers_for_agent_with_analysis(text, llm_analysis)
                    
                    # Show LLM analysis details
                    print(f"   📊 LLM Analysis:")
                    print(f"      Sentiment: {llm_analysis.get('sentiment', 'unknown')}")
                    print(f"      Emotion: {llm_analysis.get('emotion', 'unknown')}")
                    print(f"      Risk Patterns: {llm_analysis.get('risk_patterns', [])}")
                    
                    # Show service indicators
                    offer_indicators = llm_analysis.get('offer_indicators', {})
                    service_usage = offer_indicators.get('service_usage', {})
                    service_removal = offer_indicators.get('service_removal_interest', {})
                    budget_concern = offer_indicators.get('budget_concern_level', 'none')
                    
                    print(f"      Service Usage: {service_usage}")
                    print(f"      Removal Interest: {service_removal}")
                    print(f"      Budget Concern: {budget_concern}")
                    
                    # Show offer results
                    accepted = [o for o in offers if o.get('accepted', True)]
                    rejected = [o for o in offers if not o.get('accepted', True)]
                    persistent = [o for o in rejected if o.get('persistent_rejection', False)]
                    new_rejected = [o for o in rejected if not o.get('persistent_rejection', False)]
                    
                    print(f"\n   📋 Offer Results:")
                    print(f"      Total: {len(offers)} | Accepted: {len(accepted)} | Rejected: {len(rejected)} ({len(persistent)} persistent)")
                    
                    if accepted:
                        print(f"   ✅ Accepted Offers:")
                        for j, offer in enumerate(accepted[:3], 1):
                            price_str = f"${offer['price_delta']:+}" if offer['price_delta'] != 0 else "No cost"
                            print(f"      {j}. {offer['title']} - {price_str}/month")
                    
                    if new_rejected:
                        print(f"   🆕 Newly Rejected:")
                        for j, offer in enumerate(new_rejected[:2], 1):
                            reason = offer.get('rejection_reason', 'No reason')[:60]
                            print(f"      {j}. {offer['title']} - {reason}...")
                    
                    if persistent:
                        print(f"   📋 Persistent Rejections: {len(persistent)} offers remain rejected")
                
                else:
                    print(f"   ❌ LLM analysis failed: {llm_analysis['error']}")
            
            print("=" * 80)
        
        else:  # Agent
            last_agent_message = text
    
    # Final summary
    print(f"\n🏁 FINAL LLM ANALYSIS SUMMARY")
    print(f"   Final Churn Score: {scorer.current_score:.1f}/100")
    print(f"   Score Change: {scorer.current_score - 50.0:+.1f}")
    print(f"   Customer Messages: {customer_count}")
    
    # Show conversation context
    if hasattr(scorer, 'conversation_context'):
        context = scorer.conversation_context
        print(f"   💰 Prices Mentioned: {context.mentioned_prices}")
        print(f"   📺 Unused Services: {context.unused_services}")
        print(f"   🏃 Competitors: {context.competitor_mentions}")
    
    # Show persistent rejected offers
    if hasattr(scorer, 'llm_rejected_offers') and scorer.llm_rejected_offers:
        print(f"\n🚫 PERSISTENT LLM REJECTIONS ({len(scorer.llm_rejected_offers)} offers):")
        for offer_id, reason in scorer.llm_rejected_offers.items():
            print(f"   {offer_id}: {reason[:80]}...")

if __name__ == "__main__":
    test_llm_churn_and_offers()