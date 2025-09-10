import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import requests
import os
import re

@dataclass
class Offer:
    """Represents a customer offer with metadata for filtering"""
    offer_id: str
    title: str
    description: str
    price_delta: float  # Change from current price (+/- amount)
    product_types: List[str]  # ["Internet", "TV", "Mobile", "Bundle"]
    features: List[str]
    contract_type: str  # "1yr", "2yr", "flexible", "none"
    discount_eligible: bool
    retention_offer: bool  # True if this is a retention/churn prevention offer
    category: str  # "upgrade", "discount", "service", "bonus", "family"
    value_proposition: str  # "$45/month value", "Save $32.50"
    urgency: str  # "Limited Time", "Apply Now"
    relevance_score: float  # 0-100, will be dynamically calculated
    base_priority: int  # 1-5, higher = more important
    min_churn_score: float  # Minimum churn score to show this offer
    max_churn_score: float  # Maximum churn score to show this offer

@dataclass
class ConversationIntent:
    """Represents extracted intent from customer conversation"""
    intent: str  # "cost_concern", "service_issue", "competitor_mention", etc.
    entities: List[str]  # ["TV", "Internet", "Mobile"]
    confidence: float  # 0-1
    urgency: str  # "low", "medium", "high"
    emotional_state: str  # "frustrated", "neutral", "satisfied"

class DynamicOfferEngine:
    """Dynamic offer filtering and recommendation engine"""
    
    def __init__(self, embedding_model: SentenceTransformer = None):
        self.embedding_model = embedding_model or SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize intent classification via Hugging Face API
        try:
            print("Setting up intent classification via Hugging Face API...")
            self.hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            self.intent_api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            
            if self.hf_token:
                print("Intent classifier API ready!")
                self.intent_classifier = True
            else:
                print("Warning: HUGGINGFACEHUB_API_TOKEN not found, intent classification disabled")
                self.intent_classifier = None
        except Exception as e:
            print(f"Could not setup intent classifier API: {e}")
            self.intent_classifier = None
        
        # Define offer catalog with rich metadata
        self.offer_catalog = self._initialize_offer_catalog()
        
        # Intent classification labels
        self.intent_labels = [
            "cost_concern",
            "service_dissatisfaction", 
            "competitor_mention",
            "feature_request",
            "billing_complaint",
            "technical_issue",
            "cancellation_threat",
            "satisfaction_expressed"
        ]
        
        # Entity extraction patterns
        self.product_entities = {
            "TV": ["tv", "television", "cable", "channels", "streaming", "watch"],
            "Internet": ["internet", "wifi", "broadband", "connection", "speed", "data"],
            "Mobile": ["mobile", "phone", "cellular", "call", "text", "minutes"],
            "Bundle": ["package", "bundle", "everything", "all services"]
        }
        
        # Precompute entity embeddings for better extraction
        self._precompute_entity_embeddings()

        def _initialize_offer_catalog(self) -> List[Offer]:
            """Initialize the offer catalog with diverse offers"""
            return [
                Offer(
                    offer_id="BB+PKG_$240TVplus_MOB",
                    title="BB+PKG_$240TVplus_MOB",
                    description="Bundle with Internet, TV, and Mobile for $240/month",
                    price_delta=240.0,
                    product_types=["Internet", "TV", "Mobile"],
                    features=["Bundle", "Triple Play"],
                    contract_type="monthly",
                    discount_eligible=False,
                    retention_offer=False,
                    category="bundle",
                    value_proposition="All-in-one connectivity",
                    urgency="Available Now",
                    relevance_score=0.0,
                    base_priority=3,
                    min_churn_score=0.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="BB_$1851gig_TVMOB_CSpourPkg_1yr",
                    title="BB_$1851gig_TVMOB_CSpourPkg_1yr",
                    description="Internet, TV, and Mobile bundle at 1 Gbps speed for $185/month (1-year contract)",
                    price_delta=185.0,
                    product_types=["Internet", "TV", "Mobile"],
                    features=["Bundle", "1 Gbps Speed"],
                    contract_type="1-year",
                    discount_eligible=True,
                    retention_offer=False,
                    category="bundle",
                    value_proposition="High-speed triple play at lower cost",
                    urgency="Limited Availability",
                    relevance_score=0.0,
                    base_priority=4,
                    min_churn_score=0.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="Pricelock_$160TVbasic_MOB_2yr",
                    title="Pricelock_$160TVbasic_MOB_2yr",
                    description="Basic TV and Mobile package for $160/month with 2-year price lock",
                    price_delta=160.0,
                    product_types=["Internet", "TV", "Mobile"],
                    features=["Price Lock", "Bundle"],
                    contract_type="2-year",
                    discount_eligible=True,
                    retention_offer=False,
                    category="pricelock",
                    value_proposition="Stable pricing for 2 years",
                    urgency="Sign Up Now",
                    relevance_score=0.0,
                    base_priority=4,
                    min_churn_score=20.0,
                    max_churn_score=80.0
                ),
                
                Offer(
                    offer_id="BB+PKG_$260TVSmartHome_MOB",
                    title="BB+PKG_$260TVSmartHome_MOB",
                    description="Internet, TV, Mobile, and Smart Home features for $260/month",
                    price_delta=260.0,
                    product_types=["Internet", "TV", "Mobile", "SmartHome"],
                    features=["Smart Home", "Bundle"],
                    contract_type="monthly",
                    discount_eligible=False,
                    retention_offer=False,
                    category="premium",
                    value_proposition="Enhanced living with smart home add-ons",
                    urgency="Premium Offer",
                    relevance_score=0.0,
                    base_priority=2,
                    min_churn_score=0.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="BB+PKG_$260Mobile_Voice_MOB",
                    title="BB+PKG_$260Mobile_Voice_MOB",
                    description="Internet, Mobile, and Voice bundle for $260/month",
                    price_delta=260.0,
                    product_types=["Internet", "Mobile", "Voice"],
                    features=["Voice Add-on", "Bundle"],
                    contract_type="monthly",
                    discount_eligible=False,
                    retention_offer=False,
                    category="bundle",
                    value_proposition="Stay connected everywhere",
                    urgency="Standard Offer",
                    relevance_score=0.0,
                    base_priority=2,
                    min_churn_score=0.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="$240Data_MOB_2gigabitpkg_2yr",
                    title="$240Data_MOB_2gigabitpkg_2yr",
                    description="Internet and Mobile package with 2 Gbps speed for $240/month (2-year contract)",
                    price_delta=240.0,
                    product_types=["Internet", "Mobile"],
                    features=["2 Gbps Speed", "Bundle"],
                    contract_type="2-year",
                    discount_eligible=True,
                    retention_offer=False,
                    category="pricelock",
                    value_proposition="Ultra-fast speed at stable price",
                    urgency="Sign Up Now",
                    relevance_score=0.0,
                    base_priority=4,
                    min_churn_score=10.0,
                    max_churn_score=90.0
                ),
                
                Offer(
                    offer_id="Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                    title="Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                    description="Internet, Mobile, and Voice performance package for $230/month with 2-year price lock",
                    price_delta=230.0,
                    product_types=["Internet", "Mobile", "Voice"],
                    features=["Performance Bundle", "Price Lock"],
                    contract_type="2-year",
                    discount_eligible=True,
                    retention_offer=False,
                    category="pricelock",
                    value_proposition="Reliable service at fixed rate",
                    urgency="Lock Your Price",
                    relevance_score=0.0,
                    base_priority=4,
                    min_churn_score=20.0,
                    max_churn_score=80.0
                ),
                
                Offer(
                    offer_id="$2000Data_MOB_gigabitExtraPkg_2yr",
                    title="$2000Data_MOB_gigabitExtraPkg_2yr",
                    description="Internet and Mobile gigabit extra package for $2000/month (2-year contract)",
                    price_delta=2000.0,
                    product_types=["Internet", "Mobile"],
                    features=["Gigabit Extra", "Bundle"],
                    contract_type="2-year",
                    discount_eligible=False,
                    retention_offer=False,
                    category="premium",
                    value_proposition="Extra high capacity gigabit package",
                    urgency="Exclusive Plan",
                    relevance_score=0.0,
                    base_priority=1,
                    min_churn_score=0.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="BB_$150Mobile_UNLTplus_2yr",
                    title="BB_$150Mobile_UNLTplus_2yr",
                    description="Internet and Mobile unlimited plus package for $150/month (2-year contract)",
                    price_delta=150.0,
                    product_types=["Internet", "Mobile"],
                    features=["Unlimited Data", "Bundle"],
                    contract_type="2-year",
                    discount_eligible=True,
                    retention_offer=False,
                    category="value",
                    value_proposition="Affordable unlimited connectivity",
                    urgency="Great Value",
                    relevance_score=0.0,
                    base_priority=5,
                    min_churn_score=30.0,
                    max_churn_score=100.0
                ),
                
                Offer(
                    offer_id="Bundle_$140Internet+Mobile_2yr",
                    title="Bundle_$140Internet+Mobile_2yr",
                    description="Internet and Mobile bundle for $140/month (2-year contract)",
                    price_delta=140.0,
                    product_types=["Internet", "Mobile"],
                    features=["Bundle", "Contract Savings"],
                    contract_type="2-year",
                    discount_eligible=True,
                    retention_offer=True,
                    category="discount",
                    value_proposition="Lowest cost bundle option",
                    urgency="Sign Up Now",
                    relevance_score=0.0,
                    base_priority=5,
                    min_churn_score=50.0,
                    max_churn_score=100.0
                )
            ]


        def _precompute_entity_embeddings(self):
            """Precompute embeddings for product entities"""
            print("Computing entity embeddings...")
            self.entity_embeddings = {}
            
            for entity, keywords in self.product_entities.items():
                # Create diverse examples for each entity
                examples = keywords + [
                    f"I need better {entity.lower()}",
                    f"Problems with my {entity.lower()}",
                    f"My {entity.lower()} service",
                    f"{entity.lower()} is not working"
                ]
                embeddings = self.embedding_model.encode(examples)
                self.entity_embeddings[entity] = embeddings

        def extract_conversation_intent(self, customer_text: str, churn_score: float) -> ConversationIntent:
            """Extract intent and entities from customer conversation"""
            
            # Extract intent using Hugging Face API
            intent = "neutral"
            confidence = 0.5
            
            if self.intent_classifier and self.hf_token:
                try:
                    headers = {"Authorization": f"Bearer {self.hf_token}"}
                    payload = {
                        "inputs": customer_text,
                        "parameters": {
                            "candidate_labels": self.intent_labels
                        }
                    }
                    
                    response = requests.post(self.intent_api_url, headers=headers, json=payload)
                    result = response.json()
                    
                    if 'labels' in result and 'scores' in result:
                        intent = result['labels'][0]
                        confidence = result['scores'][0]
                    
                except Exception as e:
                    print(f"Intent classification API error: {e}")
            
            # Extract entities using embedding similarity
            entities = self._extract_entities(customer_text)
            
            # Determine urgency based on churn score and intent
            urgency = "low"
            if churn_score > 70 or intent in ["cancellation_threat", "competitor_mention"]:
                urgency = "high"
            elif churn_score > 50 or intent in ["cost_concern", "service_dissatisfaction"]:
                urgency = "medium"
            
            # Determine emotional state
            emotional_state = "neutral"
            if churn_score > 60:
                emotional_state = "frustrated"
            elif intent == "satisfaction_expressed":
                emotional_state = "satisfied"
            
            return ConversationIntent(
                intent=intent,
                entities=entities,
                confidence=confidence,
                urgency=urgency,
                emotional_state=emotional_state
            )

        def _extract_entities(self, text: str) -> List[str]:
            """Extract product entities using embedding similarity"""
            text_embedding = self.embedding_model.encode([text])[0]
            detected_entities = []
            
            threshold = 0.3  # Similarity threshold for entity detection
            
            for entity, entity_embeddings in self.entity_embeddings.items():
                # Calculate similarity with entity examples
                similarities = np.dot(entity_embeddings, text_embedding) / (
                    np.linalg.norm(entity_embeddings, axis=1) * np.linalg.norm(text_embedding)
                )
                
                max_similarity = np.max(similarities)
                if max_similarity > threshold:
                    detected_entities.append(entity)
            
            return detected_entities

        def filter_offers(self, intent: ConversationIntent, churn_score: float, 
                        churn_delta: float, previous_offers: List[str] = None) -> List[Offer]:
            """Filter and rank offers based on conversation context and churn score"""
            
            print(f"\n--- Offer Filtering ---")
            print(f"Intent: {intent.intent} (confidence: {intent.confidence:.2f})")
            print(f"Entities: {intent.entities}")
            print(f"Churn Score: {churn_score}, Delta: {churn_delta}")
            print(f"Urgency: {intent.urgency}, Emotion: {intent.emotional_state}")
            
            # Start with all offers
            filtered_offers = self.offer_catalog.copy()
            
            # Rule 1: Filter by churn score range
            filtered_offers = [
                offer for offer in filtered_offers
                if offer.min_churn_score <= churn_score <= offer.max_churn_score
            ]
            print(f"After churn score filter: {len(filtered_offers)} offers")
            
            # Rule 2: Intent-based filtering
            filtered_offers = self._apply_intent_filters(filtered_offers, intent)
            print(f"After intent filters: {len(filtered_offers)} offers")
            
            # Rule 3: Entity-based filtering (remove irrelevant products)
            if intent.entities:
                filtered_offers = self._apply_entity_filters(filtered_offers, intent.entities)
                print(f"After entity filters: {len(filtered_offers)} offers")
            
            # Rule 4: Churn risk escalation (prioritize retention offers)
            if churn_delta >= 5.0 or churn_score > 70:
                # Boost retention offers
                for offer in filtered_offers:
                    if offer.retention_offer:
                        offer.base_priority += 2
                print("Applied churn escalation boost to retention offers")
            
            # Rule 5: Remove previously shown offers (avoid repetition)
            if previous_offers:
                filtered_offers = [
                    offer for offer in filtered_offers 
                    if offer.offer_id not in previous_offers
                ]
                print(f"After removing previous offers: {len(filtered_offers)} offers")
            
            # Calculate relevance scores and rank
            scored_offers = self._calculate_relevance_scores(filtered_offers, intent, churn_score)
            
            # Sort by relevance score (desc) and priority (desc)
            ranked_offers = sorted(
                scored_offers, 
                key=lambda x: (x.relevance_score, x.base_priority), 
                reverse=True
            )
            
            print(f"Final ranked offers: {len(ranked_offers)}")
            for i, offer in enumerate(ranked_offers[:3]):  # Show top 3
                print(f"  {i+1}. {offer.title} (relevance: {offer.relevance_score:.1f}, priority: {offer.base_priority})")
            
            return ranked_offers

        def _apply_intent_filters(self, offers: List[Offer], intent: ConversationIntent) -> List[Offer]:
            """Apply filtering rules based on conversation intent"""
            
            if intent.intent == "cost_concern":
                # Remove expensive upgrades, prioritize discounts and downgrades
                return [
                    offer for offer in offers 
                    if offer.price_delta <= 0 or offer.discount_eligible
                ]
            
            elif intent.intent == "service_dissatisfaction":
                # Focus on service improvement offers
                return [
                    offer for offer in offers
                    if offer.category in ["service", "upgrade"] or offer.retention_offer
                ]
            
            elif intent.intent == "competitor_mention":
                # Show competitive offers and retention deals
                return [
                    offer for offer in offers
                    if offer.category in ["competitive", "discount"] or offer.retention_offer
                ]
            
            elif intent.intent == "cancellation_threat":
                # Emergency retention offers only
                return [
                    offer for offer in offers
                    if offer.retention_offer and offer.base_priority >= 4
                ]
            
            elif intent.intent == "satisfaction_expressed":
                # Can show upsell offers
                return [
                    offer for offer in offers
                    if not offer.retention_offer or offer.category in ["upgrade", "family"]
                ]
            
            # Default: return all offers
            return offers

        def _apply_entity_filters(self, offers: List[Offer], entities: List[str]) -> List[Offer]:
            """Filter offers based on mentioned product entities"""
            
            relevant_offers = []
            
            for offer in offers:
                # Check if offer is relevant to mentioned entities
                offer_relevant = False
                
                # If customer mentioned specific products, prefer offers for those products
                for entity in entities:
                    if entity in offer.product_types:
                        offer_relevant = True
                        break
                
                # Special case: if customer mentions TV negatively (wants to remove it)
                # and offer doesn't include TV, it's relevant
                if "TV" in entities and "TV" not in offer.product_types:
                    offer_relevant = True
                
                # Bundle offers are always somewhat relevant
                if "Bundle" in offer.product_types:
                    offer_relevant = True
                
                if offer_relevant:
                    relevant_offers.append(offer)
            
            return relevant_offers if relevant_offers else offers  # Fallback to all offers

        def _calculate_relevance_scores(self, offers: List[Offer], intent: ConversationIntent, 
                                    churn_score: float) -> List[Offer]:
            """Calculate relevance scores for offers based on context"""
            
            for offer in offers:
                score = 50.0  # Base score
                
                # Intent matching bonus
                intent_bonus = {
                    "cost_concern": 30 if offer.price_delta <= 0 else -20,
                    "service_dissatisfaction": 25 if offer.category in ["service", "upgrade"] else 0,
                    "competitor_mention": 35 if offer.category == "competitive" else 0,
                    "cancellation_threat": 40 if offer.retention_offer else -30,
                    "satisfaction_expressed": 20 if not offer.retention_offer else -10
                }
                score += intent_bonus.get(intent.intent, 0)
                
                # Churn score bonus
                if churn_score > 70 and offer.retention_offer:
                    score += 25
                elif churn_score < 30 and not offer.retention_offer:
                    score += 15
                
                # Entity relevance bonus
                if intent.entities:
                    entity_match = any(entity in offer.product_types for entity in intent.entities)
                    if entity_match:
                        score += 20
                
                # Priority bonus
                score += offer.base_priority * 5
                
                # Confidence penalty (lower confidence = lower scores)
                score *= intent.confidence
                
                # Ensure score is in valid range
                offer.relevance_score = max(0, min(100, score))
            
            return offers

        def get_recommended_offers(self, customer_text: str, churn_score: float, 
                                churn_delta: float, previous_offers: List[str] = None,
                                max_offers: int = 3) -> Dict:
            """Main entry point for getting offer recommendations"""
            
            # Extract conversation intent
            intent = self.extract_conversation_intent(customer_text, churn_score)
            
            # Filter and rank offers
            ranked_offers = self.filter_offers(intent, churn_score, churn_delta, previous_offers)
            
            # Return top offers with metadata
            top_offers = ranked_offers[:max_offers]
            
            return {
                "intent": asdict(intent),
                "offers": [asdict(offer) for offer in top_offers],
                "total_available": len(ranked_offers),
                "recommendation_context": {
                    "churn_score": churn_score,
                    "churn_delta": churn_delta,
                    "primary_filter": intent.intent,
                    "entities_detected": intent.entities,
                    "urgency": intent.urgency
                }
            }

# Example usage and testing
def test_offer_engine():
    """Test the offer engine with sample conversations"""
    
    engine = DynamicOfferEngine()
    
    test_cases = [
        {
            "text": "My bill is too expensive, I can't afford it anymore",
            "churn_score": 75.0,
            "churn_delta": 10.0
        },
        {
            "text": "I don't really watch TV much, maybe I should cancel it",
            "churn_score": 45.0,
            "churn_delta": 5.0
        },
        {
            "text": "T-Mobile has much better prices than you guys",
            "churn_score": 80.0,
            "churn_delta": 15.0
        },
        {
            "text": "Thanks for helping me, that sounds like a good solution",
            "churn_score": 25.0,
            "churn_delta": -10.0
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {case['text']}")
        print(f"{'='*60}")
        
        result = engine.get_recommended_offers(
            customer_text=case['text'],
            churn_score=case['churn_score'],
            churn_delta=case['churn_delta']
        )
        
        print(f"Detected Intent: {result['intent']['intent']}")
        print(f"Entities: {result['intent']['entities']}")
        print(f"Urgency: {result['intent']['urgency']}")
        print(f"\nTop Offers:")
        for j, offer in enumerate(result['offers']):
            print(f"  {j+1}. {offer['title']} (relevance: {offer['relevance_score']:.1f})")
            print(f"     {offer['description']}")
            print(f"     Value: {offer['value_proposition']}")

if __name__ == "__main__":
    test_offer_engine() 