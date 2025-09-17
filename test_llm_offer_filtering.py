#!/usr/bin/env python3

from simple_churn_scorer import SimpleChurnScorer

def test_llm_offer_filtering():
    """Test LLM-based offer filtering integration with simple churn scorer"""
    
    print("🔍 TESTING LLM OFFER FILTERING INTEGRATION")
    print("=" * 70)
    
    # Initialize scorer
    scorer = SimpleChurnScorer()
    
    # Test messages that should trigger different offer filtering
    test_cases = [
        {
            "customer": "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for.",
            "description": "TV usage concern + billing complaint"
        },
        {
            "customer": "Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper.",
            "description": "Competitor mention + price sensitivity"
        },
        {
            "customer": "Alright, let's go with that then. Please make sure the new charges reflect next month.",
            "description": "Positive resolution"
        }
    ]
    
    print("🧠 RULE-BASED OFFER FILTERING (Current System)")
    print("=" * 70)
    
    # Test with rule-based filtering (default)
    scorer.use_llm_offer_filtering = False
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 TEST CASE {i}: {test_case['description']}")
        print(f"Customer: {test_case['customer']}")
        print("-" * 50)
        
        offers = scorer.get_offers_for_agent(test_case['customer'])
        
        # Show accepted offers
        accepted_offers = [o for o in offers if o.get('accepted', True)]
        rejected_offers = [o for o in offers if not o.get('accepted', True)]
        
        print(f"✅ ACCEPTED OFFERS ({len(accepted_offers)}):")
        for j, offer in enumerate(accepted_offers[:3], 1):  # Show top 3
            print(f"   {j}. {offer['title']} - ${offer['price_delta']}/month")
            print(f"      Products: {', '.join(offer.get('product_types', []))}")
        
        if rejected_offers:
            print(f"\n❌ REJECTED OFFERS ({len(rejected_offers)}):")
            for j, offer in enumerate(rejected_offers[:2], 1):  # Show top 2
                print(f"   {j}. {offer['title']} - ${offer['price_delta']}/month")
                print(f"      Reason: {offer.get('rejection_reason', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("🤖 LLM-BASED OFFER FILTERING (New System)")
    print("=" * 70)
    
    # Test with LLM filtering
    scorer.use_llm_offer_filtering = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 TEST CASE {i}: {test_case['description']}")
        print(f"Customer: {test_case['customer']}")
        print("-" * 50)
        
        offers = scorer.get_offers_for_agent(test_case['customer'])
        
        # Show accepted offers
        accepted_offers = [o for o in offers if o.get('accepted', True)]
        rejected_offers = [o for o in offers if not o.get('accepted', True)]
        
        print(f"✅ LLM FILTERED OFFERS ({len(accepted_offers)}):")
        for j, offer in enumerate(accepted_offers[:3], 1):  # Show top 3
            print(f"   {j}. {offer['title']} - ${offer['price_delta']}/month")
            print(f"      Products: {', '.join(offer.get('product_types', []))}")
            if 'llm_explanation' in offer:
                print(f"      LLM Logic: {offer['llm_explanation'][:100]}...")
        
        if rejected_offers:
            print(f"\n❌ LLM REJECTED OFFERS ({len(rejected_offers)}):")
            for j, offer in enumerate(rejected_offers[:2], 1):  # Show top 2
                print(f"   {j}. {offer['title']} - ${offer['price_delta']}/month")
                print(f"      Reason: {offer.get('rejection_reason', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("✅ OFFER FILTERING INTEGRATION TEST COMPLETE")
    
    print(f"\nℹ️  This test compares rule-based vs LLM-based offer filtering")
    print(f"ℹ️  LLM filtering uses sophisticated conversation analysis")
    print(f"ℹ️  Both methods can be controlled independently:")
    print(f"    • scorer.use_llm_offer_filtering = True/False")
    print(f"    • scorer.use_llm_indicators = True/False")

if __name__ == "__main__":
    test_llm_offer_filtering() 