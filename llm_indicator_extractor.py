import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from huggingface_hub import InferenceClient

@dataclass
class RiskIndicators:
    billing_complaint: Dict[str, any]
    competitor_mention: Dict[str, any]
    service_frustration: Dict[str, any]
    process_frustration: Dict[str, any]
    positive_resolution: Dict[str, any]
    
@dataclass
class OfferIndicators:
    price_sensitivity: Dict[str, any]
    service_usage_patterns: Dict[str, any]
    value_preferences: Dict[str, any]
    urgency_indicators: Dict[str, any]

@dataclass
class ConversationIndicators:
    risk_indicators: RiskIndicators
    offer_indicators: OfferIndicators
    confidence_score: float
    extraction_timestamp: datetime

class LLMIndicatorExtractor:
    def __init__(self):
        # HuggingFace setup instead of Ollama
        
        self.client = InferenceClient(token=self.hf_token)
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # Fallback to Ollama if needed (commented out)
        # self.ollama_api_url = "http://localhost:11434/api/generate"
        # self.model_name = "llama3.2:3b"
        
        # Sample customer profile (hardcoded as requested)
        self.customer_profile = {
            "name": "Megan Hazelwood",
            "current_mrc": 200,
            "previous_mrc": 180,
            "tenure_months": 18,
            "current_plan": "$200Data_TV_MOB_200MBPkg_2yr",
            "services": ["Internet", "TV", "Mobile"]
        }
        
        # Global filtered offers (persists across conversation)
        self.global_filtered_offers = None
        
        # Risk pattern categories from existing rule-based system
        self.risk_patterns = [
            "billing_complaint",
            "competitor_mention", 
            "service_frustration",
            "process_frustration",
            "positive_resolution"
        ]
        
        # Emotion categories from existing system
        self.emotion_categories = [
            "anger", "frustration", "confusion", "relief", 
            "satisfaction", "fear", "disappointment", "neutral"
        ]
        
        # Sentiment scale
        self.sentiment_scale = ["very_negative", "negative", "neutral", "positive", "very_positive"]
        
        # Initial 10 offer catalog from simple_churn_scorer.py
        self.initial_offers = [
            {
                "offer_id": "BB+PKG_$240TVplus_MOB",
                "title": "BB+PKG_$240TVplus_MOB",
                "description": "Enhanced TV Plus Bundle with Premium Channels\nInternet, TV, Mobile",
                "value_proposition": "Premium entertainment package",
                "price_delta": 240,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "standard",
                "retention_offer": False,
                "category": "bundle",
                "priority": 5
            },
            {
                "offer_id": "BB_$1851gig_TVMOB_CSpourPkg_1yr",
                "title": "BB_$1851gig_TVMOB_CSpourPkg_1yr", 
                "description": "1 Gig Internet plan with TV and Mobile bundle for 1 year\nInternet, TV, Mobile",
                "value_proposition": "High-speed triple play",
                "price_delta": 185,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "1_year",
                "retention_offer": False,
                "category": "bundle",
                "priority": 3
            },
            {
                "offer_id": "Pricelock_$160TVbasic_MOB_2yr",
                "title": "Pricelock_$160TVbasic_MOB_2yr",
                "description": "Basic TV and Mobile bundle with Internet access, price locked for 2 years\nInternet, TV, Mobile",
                "value_proposition": "Price stability guarantee",
                "price_delta": 160,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "2_year",
                "retention_offer": True,
                "category": "bundle",
                "priority": 2
            },
            {
                "offer_id": "BB+PKG_$260TVSmartHome_MOB",
                "title": "BB+PKG_$260TVSmartHome_MOB",
                "description": "Smart Home Integration with IoT Device Management\nInternet, TV, Mobile, Smarthome",
                "value_proposition": "Smart home automation",
                "price_delta": 260,
                "product_types": ["Internet", "TV", "Mobile", "SmartHome"],
                "contract_type": "standard",
                "retention_offer": False,
                "category": "bundle",
                "priority": 6
            },
            {
                "offer_id": "BB+PKG_$260Mobile_Voice_MOB",
                "title": "BB+PKG_$260Mobile_Voice_MOB",
                "description": "Premium Voice Services with Enhanced Mobile Features\nInternet, Mobile, Voice",
                "value_proposition": "Premium communication suite",
                "price_delta": 260,
                "product_types": ["Internet", "Mobile", "Voice"],
                "contract_type": "standard",
                "retention_offer": False,
                "category": "bundle",
                "priority": 7
            },
            {
                "offer_id": "$240Data_MOB_2gigabitpkg_2yr",
                "title": "$240Data_MOB_2gigabitpkg_2yr",
                "description": "Ultra-Fast 2 Gigabit Data Package for Power Users\nInternet, Mobile",
                "value_proposition": "Ultra-high speed connectivity",
                "price_delta": 240,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "retention_offer": False,
                "category": "bundle",
                "priority": 4
            },
            {
                "offer_id": "Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                "title": "Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                "description": "High-Performance Bundle with 2-Year Price Guarantee\nInternet, Mobile, Voice",
                "value_proposition": "Performance with price protection",
                "price_delta": 230,
                "product_types": ["Internet", "Mobile", "Voice"],
                "contract_type": "2_year",
                "retention_offer": True,
                "category": "bundle",
                "priority": 3
            },
            {
                "offer_id": "$2000Data_MOB_gigabitExtraPkg_2yr",
                "title": "$2000Data_MOB_gigabitExtraPkg_2yr",
                "description": "Enterprise-Grade Gigabit Extra with Premium Features\nInternet, Mobile",
                "value_proposition": "Enterprise-level performance",
                "price_delta": 200,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "retention_offer": False,
                "category": "bundle",
                "priority": 8
            },
            {
                "offer_id": "BB_$150Mobile_UNLTplus_2yr",
                "title": "BB_$150Mobile_UNLTplus_2yr",
                "description": "Unlimited Plus Package with Enhanced Data Allowances\nInternet, Mobile",
                "value_proposition": "Unlimited data value",
                "price_delta": 150,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "retention_offer": False,
                "category": "bundle",
                "priority": 2
            },
            {
                "offer_id": "Bundle_$140Internet+Mobile_2yr",
                "title": "Bundle_$140Internet+Mobile_2yr",
                "description": "Essential Dual Bundle with 2-Year Contract Savings\nInternet, Mobile",
                "value_proposition": "Essential value bundle",
                "price_delta": 140,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "retention_offer": True,
                "category": "bundle",
                "priority": 1
            }
        ]

    def extract_indicators(self, customer_message: str, agent_message: str = "") -> Optional[ConversationIndicators]:
        """
        Extract risk and offer indicators from customer conversation using LLM
        """
        try:
            # Prepare the conversation context
            conversation_context = self._prepare_conversation_context(customer_message, agent_message)
            
            # Generate the comprehensive prompt
            prompt = self._generate_extraction_prompt(customer_message, conversation_context)
            
            # Call LLM for indicator extraction
            llm_response = self._call_ollama_llm(prompt)
            
            if not llm_response:
                return None
            
            # Parse and validate the response
            indicators = self._parse_llm_response(llm_response)
            
            return indicators
            
        except Exception as e:
            print(f"Error extracting indicators: {e}")
            return None
    
    def _prepare_conversation_context(self, customer_message: str, agent_message: str = "") -> str:
        """Prepare conversation context for LLM analysis"""
        context_parts = []
        
        if agent_message.strip():
            context_parts.append(f"Agent: {agent_message.strip()}")
        
        context_parts.append(f"Customer: {customer_message.strip()}")
        
        return "\n".join(context_parts)
    
    def _generate_extraction_prompt(self, customer_message: str, conversation_context: str = "") -> str:
        """Generate prompt for LLM to extract conversation indicators"""
        
        context_section = ""
        if conversation_context:
            context_section = f"""
CONVERSATION CONTEXT:
{conversation_context}

"""

        return f"""{context_section}CURRENT CUSTOMER MESSAGE:
"{customer_message}"

Analyze the customer's perspective and intent based on the conversation flow. Consider:
1. What the customer is responding to (from context)
2. What the customer is agreeing/disagreeing with
3. What services or changes are being discussed
4. The customer's attitude toward proposed solutions

Extract the following indicators in JSON format:

{{
    "risk_indicators": {{
        "billing_complaint_detected": false,
        "billing_complaint_evidence": "",
        "competitor_mention_detected": false,
        "competitor_mention_evidence": "",
        "service_frustration_detected": false,
        "service_frustration_evidence": "",
        "process_frustration_detected": false,
        "process_frustration_evidence": "",
        "positive_resolution_detected": false,
        "positive_resolution_evidence": ""
    }},
    "offer_indicators": {{
        "price_sensitivity_detected": false,
        "budget_concern_level": "none",
        "service_usage": {{
            "tv_usage": "unknown",
            "mobile_usage": "unknown",
            "internet_usage": "unknown"
        }},
        "service_removal_interest": {{
            "tv_removal": false,
            "mobile_removal": false,
            "internet_removal": false
        }},
        "service_dissatisfaction": [],
        "value_preference": "balanced",
        "contract_flexibility_needed": false
    }},
    "overall_sentiment": "neutral",
    "dominant_emotion": "neutral"
}}

Detection Guidelines:

RISK INDICATORS:
- competitor_mention: Look for mentions of other providers (TMobile, Verizon, etc.) in the conversation
- billing_complaint: Customer complaining about bills, charges, or pricing
- service_frustration: Customer frustrated with service quality or features
- process_frustration: Customer frustrated with company processes, support, etc.
- positive_resolution: Customer accepting solutions, expressing satisfaction

SERVICE USAGE PATTERNS:
- tv_usage: "low" if customer mentions barely watching TV, not using TV much
- mobile_usage: "low" if customer mentions limited mobile usage
- internet_usage: "low" if customer mentions limited internet usage
- "unknown" if no usage information mentioned

SERVICE REMOVAL INTEREST:
- tv_removal: TRUE if:
  * Customer agrees to remove TV when agent suggests it
  * Customer asks about removing TV
  * Customer accepts TV removal ("Alright, let's go with that" after TV removal discussion)
  * Agent mentions removing TV and customer shows interest or agreement
- Similar logic for mobile_removal and internet_removal

BUDGET CONCERNS:
- "high" if customer mentions bills being too expensive, can't afford current prices
- "medium" if customer shows price sensitivity but not extreme concern
- "none" if no price concerns mentioned

VALUE PREFERENCES:
- "price_focused" if customer primarily concerned with cost reduction
- "feature_focused" if customer wants more features/services
- "balanced" if customer considers both price and features

Consider the conversation flow - if agent suggests TV removal and customer responds positively ("Alright, let's go with that"), this indicates tv_removal interest even if not explicitly stated.

IMPORTANT: RETURN ONLY VALID JSON."""

    def _call_ollama_llm(self, prompt: str) -> Optional[str]:
        """Call HuggingFace LLM for indicator extraction instead of Ollama"""
        try:
            # Prepare messages in the format expected by HuggingFace chat completion
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=1000,  # Enough for our JSON response
                temperature=0.1   # Low temperature for consistent output
            )
            
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                print("No response from HuggingFace API")
                return None
                
        except Exception as e:
            print(f"Error calling HuggingFace LLM: {e}")
            return None
    
    def _parse_llm_response(self, llm_response: str) -> Optional[ConversationIndicators]:
        """Parse and validate LLM response into structured indicators"""
        try:
            # Clean the response - sometimes LLM adds extra text
            cleaned_response = self._clean_json_response(llm_response)
            
            print(f"\n=== CLEANED JSON ===")
            print(cleaned_response)
            print("=" * 50)
            
            # Parse JSON
            parsed = json.loads(cleaned_response)
            
            # Validate structure
            if not self._validate_response_structure_simple(parsed):
                print("Invalid response structure from LLM")
                return None
            
            # Create structured indicators with simplified format
            risk_indicators = RiskIndicators(
                billing_complaint={
                    "detected": parsed["risk_indicators"]["billing_complaint_detected"],
                    "evidence": parsed["risk_indicators"]["billing_complaint_evidence"]
                },
                competitor_mention={
                    "detected": parsed["risk_indicators"]["competitor_mention_detected"],
                    "evidence": parsed["risk_indicators"]["competitor_mention_evidence"]
                },
                service_frustration={
                    "detected": parsed["risk_indicators"]["service_frustration_detected"],
                    "evidence": parsed["risk_indicators"]["service_frustration_evidence"]
                },
                process_frustration={
                    "detected": parsed["risk_indicators"]["process_frustration_detected"],
                    "evidence": parsed["risk_indicators"]["process_frustration_evidence"]
                },
                positive_resolution={
                    "detected": parsed["risk_indicators"]["positive_resolution_detected"],
                    "evidence": parsed["risk_indicators"]["positive_resolution_evidence"]
                }
            )
            
            offer_indicators = OfferIndicators(
                price_sensitivity={
                    "detected": parsed["offer_indicators"]["price_sensitivity_detected"],
                    "budget_concern_level": parsed["offer_indicators"]["budget_concern_level"]
                },
                service_usage_patterns={
                    "tv_usage": parsed["offer_indicators"]["service_usage"]["tv_usage"],
                    "mobile_usage": parsed["offer_indicators"]["service_usage"]["mobile_usage"],
                    "internet_usage": parsed["offer_indicators"]["service_usage"]["internet_usage"],
                    "service_dissatisfaction": parsed["offer_indicators"]["service_dissatisfaction"],
                    "tv_removal_interest": parsed["offer_indicators"]["service_removal_interest"]["tv_removal"],
                    "mobile_removal_interest": parsed["offer_indicators"]["service_removal_interest"]["mobile_removal"],
                    "internet_removal_interest": parsed["offer_indicators"]["service_removal_interest"]["internet_removal"]
                },
                value_preferences={
                    "value_preference": parsed["offer_indicators"]["value_preference"],
                    "contract_flexibility_needed": parsed["offer_indicators"]["contract_flexibility_needed"]
                },
                urgency_indicators={
                    "urgency_level": "normal"  # Default since removed from prompt
                }
            )
            
            return ConversationIndicators(
                risk_indicators=risk_indicators,
                offer_indicators=offer_indicators,
                confidence_score=0.0,  # Default since not in simplified format
                extraction_timestamp=datetime.now()
            )
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {llm_response[:500]}...")
            return None
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        # Find the first { and last }
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in response")
        
        return response[start_idx:end_idx + 1]
    
    def _validate_response_structure(self, parsed: dict) -> bool:
        """Validate that the parsed response has the expected structure"""
        required_keys = ["risk_indicators", "offer_indicators"]
        
        if not all(key in parsed for key in required_keys):
            return False
        
        # Validate risk indicators
        risk_required = ["billing_complaint", "competitor_mention", "service_frustration", 
                        "process_frustration", "positive_resolution"]
        if not all(key in parsed["risk_indicators"] for key in risk_required):
            return False
        
        # Validate offer indicators  
        offer_required = ["price_sensitivity", "service_usage_patterns", "value_preferences",
                         "urgency_indicators", "contract_preferences"]
        if not all(key in parsed["offer_indicators"] for key in offer_required):
            return False
        
        return True

    def _validate_response_structure_simple(self, parsed: dict) -> bool:
        """Validate that the parsed response has the expected structure for the simplified JSON"""
        required_keys = ["risk_indicators", "offer_indicators"]
        
        if not all(key in parsed for key in required_keys):
            return False
        
        # Validate risk indicators
        risk_required = ["billing_complaint_detected", "billing_complaint_evidence",
                         "competitor_mention_detected", "competitor_mention_evidence",
                         "service_frustration_detected", "service_frustration_evidence",
                         "process_frustration_detected", "process_frustration_evidence",
                         "positive_resolution_detected", "positive_resolution_evidence"]
        if not all(key in parsed["risk_indicators"] for key in risk_required):
            return False
        
        # Validate offer indicators
        offer_required = ["price_sensitivity_detected", "budget_concern_level",
                         "service_usage", "service_removal_interest", "service_dissatisfaction",
                         "value_preference", "contract_flexibility_needed"]
        if not all(key in parsed["offer_indicators"] for key in offer_required):
            return False
        
        return True
    
    def get_detected_risk_patterns(self, indicators: ConversationIndicators) -> List[str]:
        """Extract list of detected risk patterns for compatibility with existing system"""
        detected_patterns = []
        
        risk_attrs = [
            ("billing_complaint", indicators.risk_indicators.billing_complaint),
            ("competitor_mention", indicators.risk_indicators.competitor_mention),
            ("service_frustration", indicators.risk_indicators.service_frustration),
            ("process_frustration", indicators.risk_indicators.process_frustration),
            ("positive_resolution", indicators.risk_indicators.positive_resolution)
        ]
        
        for pattern_name, pattern_data in risk_attrs:
            if pattern_data.get("detected", False):
                detected_patterns.append(pattern_name)
        
        return detected_patterns
    
    def get_offer_filtering_context(self, indicators: ConversationIndicators) -> Dict[str, any]:
        """Extract offer filtering context for dynamic offer engine"""
        context = {
            "price_sensitivity": {
                "detected": indicators.offer_indicators.price_sensitivity.get("detected", False),
                "confidence": 0.0, # Confidence is not in the simplified format
                "budget_constraints": False # This field is not in the simplified format
            },
            "service_preferences": {
                "tv_usage": indicators.offer_indicators.service_usage_patterns.get("tv_usage", "unknown"),
                "mobile_usage": indicators.offer_indicators.service_usage_patterns.get("mobile_usage", "unknown"),
                "internet_usage": indicators.offer_indicators.service_usage_patterns.get("internet_usage", "unknown"),
                "confidence": indicators.offer_indicators.service_usage_patterns.get("confidence", 0.0)
            },
            "value_drivers": {
                "price_focused": indicators.offer_indicators.value_preferences.get("price_focused", False),
                "feature_focused": indicators.offer_indicators.value_preferences.get("feature_focused", False),
                "convenience_focused": False, # This field is not in the simplified format
                "confidence": indicators.offer_indicators.value_preferences.get("confidence", 0.0)
            },
            "urgency": {
                "immediate_resolution_needed": False, # This field is not in the simplified format
                "considering_alternatives": False, # This field is not in the simplified format
                "confidence": 0.0
            },
            "contract_flexibility": {
                "prefers_flexibility": indicators.offer_indicators.contract_preferences.get("prefers_flexibility", False),
                "accepts_long_term": False, # This field is not in the simplified format
                "confidence": indicators.offer_indicators.contract_preferences.get("confidence", 0.0)
            }
        }
        
        return context

    def get_comprehensive_analysis(self, customer_message: str, agent_context: str = "") -> Dict[str, any]:
        """
        Get comprehensive analysis including sentiment, emotion, risk patterns, and filtered offers
        """
        # Extract indicators
        indicators = self.extract_indicators(customer_message, agent_context)
        
        if not indicators:
            return {
                "error": "Failed to extract indicators",
                "sentiment": "unknown",
                "emotion": "unknown",
                "risk_patterns": [],
                "filtered_offers": []
            }
        
        # Get detected risk patterns
        detected_risks = self.get_detected_risk_patterns(indicators)
        
        # Filter and rank offers based on indicators
        filtered_offers = self.filter_offers_by_indicators(indicators)
        
        return {
            "sentiment": self._get_parsed_sentiment(indicators),
            "emotion": self._get_parsed_emotion(indicators),
            "risk_patterns": detected_risks,
            "filtered_offers": filtered_offers,
            "tv_usage_analysis": self._get_tv_usage_analysis(indicators),
            "price_analysis": self._get_price_analysis(indicators)
        }
    
    def filter_offers_by_indicators(self, indicators: ConversationIndicators) -> List[Dict]:
        """Filter offers based on simple rules - no scoring, just filtering and sorting"""
        
        # Initialize offers from global filtered offers or all offers
        if self.global_filtered_offers is None:
            self.global_filtered_offers = [offer.copy() for offer in self.initial_offers]
        
        current_offers = [offer.copy() for offer in self.global_filtered_offers]
        
        # Extract indicators
        budget_concern = indicators.offer_indicators.price_sensitivity.get("budget_concern_level", "none")
        
        # Check competitor mention from RISK PATTERNS only
        competitor_mentioned = indicators.risk_indicators.competitor_mention.get("detected", False)
        
        # Service usage and removal interest
        tv_usage = indicators.offer_indicators.service_usage_patterns.get("tv_usage", "unknown")
        tv_removal_interest = indicators.offer_indicators.service_usage_patterns.get("tv_removal_interest", False)
        mobile_usage = indicators.offer_indicators.service_usage_patterns.get("mobile_usage", "unknown")
        mobile_removal_interest = indicators.offer_indicators.service_usage_patterns.get("mobile_removal_interest", False)
        internet_usage = indicators.offer_indicators.service_usage_patterns.get("internet_usage", "unknown")
        internet_removal_interest = indicators.offer_indicators.service_usage_patterns.get("internet_removal_interest", False)
        
        customer_mrc = self.customer_profile["current_mrc"]
        
        # Step 1: Remove offers based on service removal interest
        offers_to_remove = []
        for i, offer in enumerate(current_offers):
            should_remove = False
            removal_reasons = []
            
            if tv_removal_interest and "TV" in offer["product_types"]:
                should_remove = True
                removal_reasons.append("Customer wants to remove TV service")
            
            if mobile_removal_interest and "Mobile" in offer["product_types"]:
                should_remove = True
                removal_reasons.append("Customer wants to remove Mobile service")
                
            if internet_removal_interest and "Internet" in offer["product_types"]:
                should_remove = True
                removal_reasons.append("Customer wants to remove Internet service")
            
            if should_remove:
                offer["removal_reason"] = "; ".join(removal_reasons)
                offers_to_remove.append(i)
        
        # Remove offers (in reverse order to maintain indices)
        for i in reversed(offers_to_remove):
            current_offers.pop(i)
        
        # Step 2: Filter based on budget concerns and customer MRC
        if budget_concern == "high":
            # Remove offers above current MRC
            current_offers = [offer for offer in current_offers if offer["price_delta"] <= customer_mrc]
            filtering_reason = f"High budget concern: Removed offers above current MRC ${customer_mrc}"
        elif budget_concern == "medium":
            # Keep offers below customer MRC, sort higher to lower
            current_offers = [offer for offer in current_offers if offer["price_delta"] < customer_mrc]
            current_offers.sort(key=lambda x: x["price_delta"], reverse=True)
            filtering_reason = f"Medium budget concern: Offers below ${customer_mrc}, sorted higher to lower"
        else:
            filtering_reason = "No budget filtering applied"
        
        # Step 3: Handle service usage (low usage moves offers to bottom)
        main_offers = []
        low_usage_offers = []
        
        for offer in current_offers:
            is_low_usage = False
            
            if tv_usage == "low" and "TV" in offer["product_types"]:
                is_low_usage = True
                offer["low_usage_reason"] = "Customer barely watches TV"
            elif mobile_usage == "low" and "Mobile" in offer["product_types"]:
                is_low_usage = True  
                offer["low_usage_reason"] = "Customer barely uses Mobile"
            elif internet_usage == "low" and "Internet" in offer["product_types"]:
                is_low_usage = True
                offer["low_usage_reason"] = "Customer barely uses Internet"
            
            if is_low_usage:
                low_usage_offers.append(offer)
            else:
                main_offers.append(offer)
        
        # Step 4: Handle competitor mentions (sort by smallest price for retention)
        if competitor_mentioned:
            main_offers.sort(key=lambda x: x["price_delta"])
            low_usage_offers.sort(key=lambda x: x["price_delta"])
            sorting_reason = "Competitor mentioned in risk patterns: Sorted by smallest price for retention"
        else:
            sorting_reason = "No retention sorting applied"
        
        # Combine main offers and low usage offers
        final_offers = main_offers + low_usage_offers
        
        # Add filtering explanation to each offer
        for offer in final_offers:
            offer["filtering_explanation"] = []
            offer["filtering_explanation"].append(filtering_reason)
            offer["filtering_explanation"].append(sorting_reason)
            
            if hasattr(offer, 'low_usage_reason'):
                offer["filtering_explanation"].append(f"Low usage: {offer['low_usage_reason']}")
        
        # Update global filtered offers
        self.global_filtered_offers = final_offers
        
        return final_offers
    
    def _get_parsed_sentiment(self, indicators: ConversationIndicators) -> str:
        """Extract overall sentiment from indicators"""
        try:
            # Try to get sentiment from the JSON response
            return getattr(indicators, 'overall_sentiment', 'neutral')
        except:
            return 'neutral'
    
    def _get_parsed_emotion(self, indicators: ConversationIndicators) -> str:
        """Extract dominant emotion from indicators"""
        try:
            # Try to get emotion from the JSON response  
            return getattr(indicators, 'dominant_emotion', 'neutral')
        except:
            return 'neutral'
    
    def _get_tv_usage_analysis(self, indicators: ConversationIndicators) -> Dict[str, any]:
        """Provide detailed TV usage analysis"""
        tv_usage = indicators.offer_indicators.service_usage_patterns.get("tv_usage", "unknown")
        tv_removal_interest = indicators.offer_indicators.service_usage_patterns.get("tv_removal_interest", False)
        
        analysis = {
            "tv_usage_level": tv_usage,
            "removal_interest": tv_removal_interest,
            "offer_strategy": ""
        }
        
        if tv_removal_interest or tv_usage == "to_remove":
            analysis["offer_strategy"] = "REMOVE all TV offers - customer wants TV service removed"
        elif tv_usage == "unused":
            analysis["offer_strategy"] = "AVOID TV offers - customer doesn't use TV at all"
        elif tv_usage == "low":
            analysis["offer_strategy"] = "DEPRIORITIZE TV offers - customer barely watches TV"
        elif tv_usage == "high":
            analysis["offer_strategy"] = "PRIORITIZE TV offers - customer actively uses TV"
        else:
            analysis["offer_strategy"] = "NO FILTERING - TV usage unknown"
            
        return analysis
    
    def _get_price_analysis(self, indicators: ConversationIndicators) -> Dict[str, any]:
        """Provide detailed price sensitivity analysis"""
        price_sensitive = indicators.offer_indicators.price_sensitivity.get("detected", False)
        budget_concern = indicators.offer_indicators.price_sensitivity.get("budget_concern_level", "none")
        cost_reduction_priority = indicators.offer_indicators.value_preferences.get("cost_reduction_priority", False)
        
        return {
            "price_sensitive": price_sensitive,
            "budget_concern_level": budget_concern,
            "cost_reduction_priority": cost_reduction_priority,
            "recommended_price_range": self._get_recommended_price_range(budget_concern)
        }
    
    def _get_recommended_price_range(self, budget_concern: str) -> str:
        """Get recommended price range based on budget concerns"""
        if budget_concern == "high":
            return "Under $150"
        elif budget_concern == "medium":
            return "$150-$200"
        else:
            return "No specific range"

# Test function
def test_comprehensive_analysis():
    """Test the comprehensive analysis with simple offer filtering"""
    extractor = LLMIndicatorExtractor()
    
    # Test cases representing the conversation flow
    test_cases = [
        {
            "customer": "I just opened my bill and it's $200 again. Last month it was $180. Why is it going up every time?",
            "agent": "I understand that's frustrating. Let me walk through your bill to identify the changes.",
            "description": "Initial billing complaint"
        },
        {
            "customer": "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for.",
            "agent": "I see here that your promotional discount expired a month ago, and there's a rental charge for a second set-top box.",
            "description": "TV usage revelation"
        },
        {
            "customer": "Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper.",
            "agent": "I understand. Since you mentioned you barely watch TV, Do you want to remove that?",
            "description": "Competitor mention + Agent suggests TV removal"
        },
        {
            "customer": "How much would that cost me?",
            "agent": "So your new bill would be around $150/month & I can also add a 10 dollar discount if you subscribe for ebill which would make it $140 going forward.",
            "description": "Customer inquires about TV removal cost"
        },
        {
            "customer": "Alright, let's go with that then. Please make sure the new charges reflect next month.",
            "agent": "I've updated your plan and removed the rental. You'll see the changes in your next cycle. Anything else I can help you with today?",
            "description": "Customer accepts TV removal"
        }
    ]
    
    print("🔍 SIMPLE OFFER FILTERING TEST")
    print("=" * 70)
    print(f"Customer: {extractor.customer_profile['name']}")
    print(f"Current MRC: ${extractor.customer_profile['current_mrc']}")
    print(f"Previous MRC: ${extractor.customer_profile['previous_mrc']}")
    print("=" * 70)
    
    conversation_context = ""
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 TURN {i}: {test_case['description']}")
        print("-" * 50)
        
        print(f"Customer: {test_case['customer']}")
        print(f"Agent Context: {test_case['agent']}")
        
        # Build conversation context (last 4-5 turns)
        if conversation_context:
            conversation_context += f"\nAgent: {test_case['agent']}\nCustomer: {test_case['customer']}"
        else:
            conversation_context = f"Agent: {test_case['agent']}\nCustomer: {test_case['customer']}"
        
        # Keep only last 5 conversation turns (10 lines: agent + customer)
        context_lines = conversation_context.split('\n')
        if len(context_lines) > 10:  # 5 turns = 10 lines (agent + customer)
            conversation_context = '\n'.join(context_lines[-10:])
        
        # Get comprehensive analysis
        analysis = extractor.get_comprehensive_analysis(test_case['customer'], conversation_context)
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            continue
        
        # Display current turn analysis
        print(f"\n🤖 CURRENT TURN ANALYSIS:")
        print(f"   📊 SENTIMENT: {analysis['sentiment']}")
        print(f"   😊 EMOTION: {analysis['emotion']}")
        print(f"   🚨 RISK PATTERNS: {analysis['risk_patterns']}")
        
        # Service Analysis
        tv_analysis = analysis['tv_usage_analysis']
        print(f"\n   📺 SERVICE ANALYSIS:")
        print(f"      TV Usage: {tv_analysis['tv_usage_level']}")
        print(f"      TV Removal Interest: {tv_analysis['removal_interest']}")
        
        # Show filtered offers
        print(f"\n🎯 FILTERED OFFERS ({len(analysis['filtered_offers'])} total):")
        
        # Show all offers with their filtering explanations
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
        
        print("\n" + "="*70)

if __name__ == "__main__":
    test_comprehensive_analysis() 