#!/usr/bin/env python3

from simple_churn_scorer import SimpleChurnScorer

def test_llm_churn_and_offers():
    """Test LLM-based churn scoring and offer filtering with full conversation"""
    
    # Full conversation from simple_churn_scorer.py
    conversation = [
    ("Agent", "Thank you for calling us, this is Alina. Can I have your account number please?"),
    ("Customer", "Yeah, it's 75299134."),
    ("Agent", "Thanks. And just to verify the account, am I speaking with Stephanie?"),
    ("Customer", "Yes, that's me."),
    ("Agent", "Great. For your security, I've just sent a 4-digit code to your mobile ending in 6641. Could you read it back for me?"),
    ("Customer", "One sec <pause 1 sec> okay, got it. It's 4472."),
    ("Agent", "Perfect, you're verified. Let me bring up your account details. This might take just a few seconds... Do you mind holding?"),
    ("Customer", "Sure."),
    ("Agent", "Thanks for waiting, Stephanie. I have your account open. How can I help you today?"),
    ("Customer", "I'm calling to cancel my service."),
    ("Agent", "Oh, I'm certainly sorry to hear that. We'd hate to lose you. May I ask what's led you to this decision?"),
    ("Customer", "I just got an offer from Horizon for 1 Gbps internet plus their basic TV for $85 a month. That's $50 less than what I'm paying you."),
    ("Agent", "I see. $85 for a Gig plan is a very aggressive offer. I'm looking at your current bill, and I see you're at $135 for our 400 Mbps plan and the 'Platinum' TV tier. I can definitely see the price gap."),
    ("Customer", "Exactly. And frankly, I'm tired of it. I've been paying $135 forever, and every single year I have to call in and beg for some new discount. It's exhausting."),
    ("Agent", "That makes perfect sense, and I truly apologize for that experience. It should not feel that way, especially for a loyal customer like yourself. I see you've been with us for over 8 years. Let me pull up our current retention offers to see if we can find something that better fits your needs and budget. One moment."),
    ("Agent", "Okay, thanks for holding. Because you've been with us so long, I can apply a $20/mo loyalty credit. I can also waive your modem rental fee, which looks to be $14/mo. That would be for the next 6 months. That brings your bill down by about $34."),
    ("Customer", "That's better, but it's still over $100, and only for 6 months. The $85 from Horizon is for a full year. And my speeds drop to a crawl every evening around 8 PM, so I'm not even getting what I pay for."),
    ("Agent", "Understood. You're hitting on two key issues: the price and the performance. Let's tackle the speed drop first. You shouldn't be experiencing that. I'm running a diagnostic on your line now..."),
    ("Agent", "<pause 1 sec> Okay, I see your modem is an older model. Those can struggle with network congestion in the evenings. The modem rental I offered to waive would be for a free upgrade to our new gateway. That will make a significant difference in stability and managing those peak-time speeds."),
    ("Customer", "A new modem might help, but the price is still the main problem. I'm still nowhere near $85."),
    ("Agent", "I understand. Let's look at the other half of your bill: the 'Platinum' TV tier. My system shows you have over 300 channels, but your primary set-top box usage is on local news and sports. Is that accurate?"),
    ("Customer", "Yeah, honestly, I mostly stream on Netflix and Hulu. I don't even know what's in that huge TV package you have me on. I just need the local channels and maybe ESPN."),
    ("Agent", "That's a perfect opportunity then. We can downgrade you from the 'Platinum' tier to our 'Base TV' tier. It keeps all your major local channels — ABC, CBS, NBC, Fox — plus essentials like ESPN. This change alone would save you $25 per month, and that's a permanent change, not a promotion."),
    ("Customer", "Okay <pause 1 sec> so, let me get this straight. The $135 bill <pause 1 sec> minus $20 for the loyalty credit, and minus $25 for the smaller TV package?"),
    ("Agent", "That's correct. That would bring your new ongoing price to $90 per month, plus tax."),
    ("Customer", "And what about the modem? You said 6 months for the credit?"),
    ("Agent", "My apologies, let me clarify. The $25 TV downgrade is permanent. The $20 loyalty credit, I can lock that in for a full 12 months. And I'll waive the new 3.1 modem rental fee for 12 months as well. So, you'd be at $90/mo for the next year."),
    ("Customer", "$90/mo <pause 1 sec> that's very close to $85. And I get a new modem that will hopefully fix the slowdowns."),
    ("Agent", "Exactly. You get a much more stable connection, and a package that's actually built for how you watch TV, all while saving $45 a month."),
    ("Customer", "That sounds fair. Okay, Alina, you've been helpful. Let's do that. Keep the credits, change the TV, and send me the new modem."),
    ("Agent", "That's wonderful news, Stephanie. I'm so glad we found a solution. I'm processing those changes now. The TV package change is effective immediately. The new modem will ship out today and should arrive in 2-3 business days with simple self-install instructions. I'm also adding these notes and the new pricing to your account. Is there anything else at all I can help you with today?"),
    ("Customer", "No, that's everything. Thank you for your help."),
    ("Agent", "You're very welcome. Thank you for giving us the chance to keep you as a customer. Have a great rest of your day."),
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