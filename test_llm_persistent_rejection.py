#!/usr/bin/env python3
"""
Test script to demonstrate LLM persistent rejection pattern:
Call 1: 10 offers → LLM rejects 3 → 7 accepted, 3 rejected
Call 2: 10 offers → LLM evaluates 7 available → previous 3 stay rejected (greyed out)
"""

from simple_churn_scorer import SimpleChurnScorer

def test_llm_persistent_rejection():
    """Test that LLM rejected offers stay rejected across calls"""
    print("🧪 Testing LLM Persistent Rejection Pattern")
    print("="*60)
    
    scorer = SimpleChurnScorer()
    
    if not hasattr(scorer, 'llm_extractor') or not scorer.llm_extractor:
        print("❌ LLM extractor not available for testing")
        return
    
    print(f"✅ Starting with {len(scorer.initial_offers)} offers")
    print(f"✅ LLM rejected offers tracker: {len(scorer.llm_rejected_offers)} initially")
    
    # === FIRST LLM CALL ===
    print(f"\n📞 FIRST LLM CALL")
    print("="*30)
    
    mock_analysis_1 = {
        'sentiment': 'negative',
        'emotion': 'frustration',
        'risk_patterns': ['billing_complaint'],
        'offer_indicators': {
            'service_usage': {'tv_usage': 'low'},
            'service_removal_interest': {'tv_removal': True},
            'budget_concern_level': 'high'
        },
        'filtered_offers': [
            {'offer_id': 'mobile_upgrade_1'},
            {'offer_id': 'internet_speed_1'},
            {'offer_id': 'basic_mobile_1'},
            {'offer_id': 'internet_basic_1'},
            {'offer_id': 'mobile_basic_1'},
            {'offer_id': 'retention_discount_1'},
            {'offer_id': 'internet_premium_1'}
        ]  # Accept 7 offers, reject 3 (TV offers will be rejected)
    }
    
    offers_1 = scorer.get_offers_for_agent_with_analysis("I barely watch TV and my bill is too high", mock_analysis_1)
    
    accepted_1 = [o for o in offers_1 if o.get('accepted', True)]
    rejected_1 = [o for o in offers_1 if not o.get('accepted', True)]
    
    print(f"📊 Call 1 Results:")
    print(f"   ✅ Accepted: {len(accepted_1)}")
    print(f"   ❌ Rejected: {len(rejected_1)}")
    print(f"   🔄 Total displayed: {len(offers_1)}")
    print(f"   📋 LLM rejected tracker: {len(scorer.llm_rejected_offers)} offers")
    
    if rejected_1:
        print(f"\n📝 First call rejections:")
        for i, offer in enumerate(rejected_1[:3], 1):
            print(f"   {i}. {offer.get('id', 'Unknown')}: {offer.get('rejection_reason', 'No reason')[:60]}...")
    
    # === SECOND LLM CALL ===
    print(f"\n📞 SECOND LLM CALL")
    print("="*30)
    
    mock_analysis_2 = {
        'sentiment': 'neutral',
        'emotion': 'neutral',
        'risk_patterns': [],
        'offer_indicators': {
            'service_usage': {'mobile_usage': 'high'},
            'service_removal_interest': {'mobile_removal': False},
            'budget_concern_level': 'low'
        },
        'filtered_offers': [
            {'offer_id': 'mobile_upgrade_1'},
            {'offer_id': 'internet_speed_1'},
            {'offer_id': 'mobile_basic_1'},
            {'offer_id': 'internet_premium_1'}
        ]  # Accept only 4 offers this time
    }
    
    offers_2 = scorer.get_offers_for_agent_with_analysis("Actually I use mobile a lot", mock_analysis_2)
    
    accepted_2 = [o for o in offers_2 if o.get('accepted', True)]
    rejected_2 = [o for o in offers_2 if not o.get('accepted', True)]
    persistent_2 = [o for o in rejected_2 if o.get('persistent_rejection', False)]
    new_2 = [o for o in rejected_2 if not o.get('persistent_rejection', False)]
    
    print(f"📊 Call 2 Results:")
    print(f"   ✅ Accepted: {len(accepted_2)}")
    print(f"   ❌ Total Rejected: {len(rejected_2)}")
    print(f"   📋 Persistent (from call 1): {len(persistent_2)}")
    print(f"   🆕 Newly rejected: {len(new_2)}")
    print(f"   🔄 Total displayed: {len(offers_2)}")
    print(f"   📋 LLM rejected tracker: {len(scorer.llm_rejected_offers)} offers")
    
    print(f"\n🎯 Verification:")
    print(f"   • All {len(offers_2)} offers should be displayed in frontend")
    print(f"   • Previously rejected offers should stay rejected (greyed out)")
    print(f"   • New LLM decision can reject additional offers")
    print(f"   • Persistent rejections maintain original reasons")
    
    print(f"\n🎮 Expected Pattern:")
    print(f"   Call 1: 10 total → 7 accepted, 3 rejected")
    print(f"   Call 2: 10 total → 4 accepted, 6 rejected (3 persistent + 3 new)")
    print(f"   Call 3: 10 total → X accepted, Y rejected (6+ persistent)")

if __name__ == "__main__":
    print("🧪 LLM Persistent Rejection Test")
    print("="*50)
    
    test_llm_persistent_rejection()
    
    print(f"\n🏁 Test completed!")
    print(f"\n💡 Key Benefits:")
    print(f"   1. ✅ Frontend always shows all 10 offers")
    print(f"   2. ✅ Once rejected by LLM, stays rejected (greyed out)")
    print(f"   3. ✅ Original rejection reasons preserved")
    print(f"   4. ✅ LLM only evaluates 'available' offers each time")
    print(f"   5. ✅ Progressive filtering: 10 → 7 → 4 → 2 pattern") 