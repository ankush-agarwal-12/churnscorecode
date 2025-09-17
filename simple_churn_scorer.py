import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import re
import os
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    raise ImportError("VADER sentiment analyzer is required. Install with: pip install vaderSentiment")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    raise ImportError("Sentence transformers required for embeddings. Install with: pip install sentence-transformers")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: Transformers not available. Will use VADER only for sentiment analysis.")

# Import the dynamic offer engine
try:
    from dynamic_offer_engine import DynamicOfferEngine
    OFFER_ENGINE_AVAILABLE = True
except ImportError:
    OFFER_ENGINE_AVAILABLE = False
    print("Note: Dynamic offer engine not available. Offers will not be generated.")

# Import the LLM indicator extractor
try:
    from llm_indicator_extractor import LLMIndicatorExtractor
    LLM_EXTRACTOR_AVAILABLE = True
except ImportError:
    LLM_EXTRACTOR_AVAILABLE = False
    print("Note: LLM indicator extractor not available. Will use only rule-based patterns.")

@dataclass
class ChurnEvent:
    timestamp: datetime
    speaker: str
    text: str
    agent_context: str
    sentiment_score: float
    emotion_result: dict
    risk_delta: float
    cumulative_score: float
    confidence: float
    detected_patterns: list

@dataclass
class ConversationContext:
    mentioned_prices: List[float]
    unused_services: List[str]  # ['TV', 'Mobile', 'Internet']
    price_trend: str  # 'increasing', 'stable', 'decreasing'
    patterns_detected: List[str]
    sentiment_trend: str  # 'positive', 'negative', 'neutral'
    competitor_mentions: List[str]
    current_bill: Optional[float] = None
    price_ceiling: Optional[float] = None

class SimpleChurnScorer:
    def __init__(self):
        # Initialize sentiment analyzers
        if not VADER_AVAILABLE:
            raise ImportError("VADER sentiment analyzer is required")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load a more sophisticated sentiment model for customer service
        self.advanced_sentiment = None
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading advanced sentiment analysis model...")
                # Using RoBERTa fine-tuned for sentiment analysis
                self.advanced_sentiment = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                print("Advanced sentiment model loaded successfully!")
            except Exception as e:
                print(f"Could not load advanced sentiment model: {e}")
                print("Falling back to VADER only.")
        
        # Initialize embedding model
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("Sentence transformers required for embeddings")
        print("Loading embedding model...")
        # Using more powerful model for better customer service understanding
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')   
        
        # Initialize dynamic offer engine
        if OFFER_ENGINE_AVAILABLE:
            try:
                print("Initializing dynamic offer engine...")
                self.offer_engine = DynamicOfferEngine(embedding_model=self.embedding_model)
                print("Offer engine initialized successfully!")
            except Exception as e:
                print(f"Could not initialize offer engine: {e}")
                self.offer_engine = None
        else:
            self.offer_engine = None
        
        # Initialize LLM indicator extractor
        self.use_llm_indicators = False  # Flag to enable/disable LLM-based pattern detection
        self.use_llm_offer_filtering = False  # Flag to enable/disable LLM-based offer filtering
        self.llm_rejected_offers = {}  # Track offers rejected by LLM with reasons (persistent across calls)
        if LLM_EXTRACTOR_AVAILABLE:
            try:
                print("Initializing LLM indicator extractor...")
                self.llm_extractor = LLMIndicatorExtractor()
                print("LLM extractor initialized successfully!")
                print("💡 To use LLM-based pattern detection, set scorer.use_llm_indicators = True")
                print("💡 To use LLM-based offer filtering, set scorer.use_llm_offer_filtering = True")
            except Exception as e:
                print(f"Could not initialize LLM extractor: {e}")
                self.llm_extractor = None
        else:
            self.llm_extractor = None
        
        # Churn scoring parameters
        self.baseline_score = 50.0  # Starting churn risk (0-100)
        self.current_score = self.baseline_score
        self.conversation_history = []
        self.risk_events = []
        self.previous_offers = []  # Track shown offers to avoid repetition
        
        # Exponential moving average factor (higher = more weight to recent events)
        self.alpha = 0.3
        
        # Initialize conversation context for rule-based filtering
        self.conversation_context = ConversationContext(
            mentioned_prices=[],
            unused_services=[],
            price_trend='stable',
            patterns_detected=[],
            sentiment_trend='neutral',
            competitor_mentions=[]
        )
        
        # Initial 10 offer catalog from OffersPanel.tsx (Table 1)
        self.initial_offers = [
            {
                "offer_id": "BB+PKG_$240TVplus_MOB",
                "title": "BB+PKG_$240TVplus_MOB",
                "description": "Enhanced TV Plus Bundle with Premium Channels\nInternet, TV, Mobile",
                "value_proposition": "Premium entertainment package",
                "price_delta": 240,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "standard",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "medium",
                "relevance_score": 90,
                "priority": 5,
                "churn_score_min": 0,
                "churn_score_max": 100
            },
            {
                "offer_id": "BB_$1851gig_TVMOB_CSpourPkg_1yr",
                "title": "BB_$1851gig_TVMOB_CSpourPkg_1yr", 
                "description": "1 Gig Internet plan with TV and Mobile bundle for 1 year\nInternet, TV, Mobile",
                "value_proposition": "High-speed triple play",
                "price_delta": 185,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "1_year",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "high",
                "relevance_score": 95,
                "priority": 3,
                "churn_score_min": 0,
                "churn_score_max": 100
            },
            {
                "offer_id": "Pricelock_$160TVbasic_MOB_2yr",
                "title": "Pricelock_$160TVbasic_MOB_2yr",
                "description": "Basic TV and Mobile bundle with Internet access, price locked for 2 years\nInternet, TV, Mobile",
                "value_proposition": "Price stability guarantee",
                "price_delta": 160,
                "product_types": ["Internet", "TV", "Mobile"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": True,
                "category": "bundle",
                "urgency": "high",
                "relevance_score": 88,
                "priority": 2,
                "churn_score_min": 40,
                "churn_score_max": 100
            },
            {
                "offer_id": "BB+PKG_$260TVSmartHome_MOB",
                "title": "BB+PKG_$260TVSmartHome_MOB",
                "description": "Smart Home Integration with IoT Device Management\nInternet, TV, Mobile, Smarthome",
                "value_proposition": "Smart home automation",
                "price_delta": 260,
                "product_types": ["Internet", "TV", "Mobile", "SmartHome"],
                "contract_type": "standard",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "medium",
                "relevance_score": 75,
                "priority": 6,
                "churn_score_min": 0,
                "churn_score_max": 60
            },
            {
                "offer_id": "BB+PKG_$260Mobile_Voice_MOB",
                "title": "BB+PKG_$260Mobile_Voice_MOB",
                "description": "Premium Voice Services with Enhanced Mobile Features\nInternet, Mobile, Voice",
                "value_proposition": "Premium communication suite",
                "price_delta": 260,
                "product_types": ["Internet", "Mobile", "Voice"],
                "contract_type": "standard",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "low",
                "relevance_score": 70,
                "priority": 7,
                "churn_score_min": 0,
                "churn_score_max": 50
            },
            {
                "offer_id": "$240Data_MOB_2gigabitpkg_2yr",
                "title": "$240Data_MOB_2gigabitpkg_2yr",
                "description": "Ultra-Fast 2 Gigabit Data Package for Power Users\nInternet, Mobile",
                "value_proposition": "Ultra-high speed connectivity",
                "price_delta": 240,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "medium",
                "relevance_score": 80,
                "priority": 4,
                "churn_score_min": 0,
                "churn_score_max": 100
            },
            {
                "offer_id": "Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                "title": "Pricelock_$230Bperf_Voice_MOB_pkg_2yr",
                "description": "High-Performance Bundle with 2-Year Price Guarantee\nInternet, Mobile, Voice",
                "value_proposition": "Performance with price protection",
                "price_delta": 230,
                "product_types": ["Internet", "Mobile", "Voice"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": True,
                "category": "bundle",
                "urgency": "high",
                "relevance_score": 85,
                "priority": 3,
                "churn_score_min": 30,
                "churn_score_max": 100
            },
            {
                "offer_id": "$2000Data_MOB_gigabitExtraPkg_2yr",
                "title": "$2000Data_MOB_gigabitExtraPkg_2yr",
                "description": "Enterprise-Grade Gigabit Extra with Premium Features\nInternet, Mobile",
                "value_proposition": "Enterprise-level performance",
                "price_delta": 200,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "medium",
                "relevance_score": 60,
                "priority": 8,
                "churn_score_min": 0,
                "churn_score_max": 100
            },
            {
                "offer_id": "BB_$150Mobile_UNLTplus_2yr",
                "title": "BB_$150Mobile_UNLTplus_2yr",
                "description": "Unlimited Plus Package with Enhanced Data Allowances\nInternet, Mobile",
                "value_proposition": "Unlimited data value",
                "price_delta": 150,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": False,
                "category": "bundle",
                "urgency": "medium",
                "relevance_score": 85,
                "priority": 2,
                "churn_score_min": 0,
                "churn_score_max": 100
            },
            {
                "offer_id": "Bundle_$140Internet+Mobile_2yr",
                "title": "Bundle_$140Internet+Mobile_2yr",
                "description": "Essential Dual Bundle with 2-Year Contract Savings\nInternet, Mobile",
                "value_proposition": "Essential value bundle",
                "price_delta": 140,
                "product_types": ["Internet", "Mobile"],
                "contract_type": "2_year",
                "eligibility": "all",
                "retention_offer": True,
                "category": "bundle",
                "urgency": "high",
                "relevance_score": 92,
                "priority": 1,
                "churn_score_min": 20,
                "churn_score_max": 100
            }
        ]
        
        # Risk pattern examples for embedding similarity - diverse and rich examples
        self.risk_patterns = {
            "billing_complaint": {
                "examples": [
                    # Formal complaints
                    "my bill is significantly higher than expected",
                    "these charges are unreasonable and excessive",
                    "I'm being overcharged for services I don't use",
                    "the pricing structure is completely unfair",
                    "my monthly costs have increased without explanation",
                    "these fees are not justified by the service level",
                    
                    # Casual/emotional
                    "my bill is way too high",
                    "this is ridiculously expensive",
                    "I can't afford these prices anymore",
                    "you guys are ripping me off",
                    "this bill is insane",
                    "I'm paying too much for what I get",
                    
                    # Direct/short
                    "bill too high",
                    "overpriced service",
                    "too expensive",
                    "costs too much",
                    "can't pay this",
                    "price is wrong",
                    
                    # Longer explanations
                    "I've been a loyal customer for years but my bill keeps going up every month and I don't understand why",
                    "the amount I'm paying doesn't match the value I'm receiving from your service",
                    "I signed up expecting one price but now I'm being charged much more than originally quoted"
                ],
                "base_risk": 15.0
            },
            "competitor_mention": {
            "examples": [
                    "other providers have better deals",
                    "competitors are offering lower prices",
                    "another company quoted me less",
                    "different carriers have better rates",
                    "rival companies treat customers better",
                    "I'm considering switching to a different provider",
                    "I'm looking at other options available",
                    "I'm thinking about changing services",
                    "I might need to find an alternative",
                    "I'm exploring other telecommunications companies",
                    "I'm gonna switch if this doesn't get fixed",
                    "thinking of jumping ship",
                    "ready to leave for someone else",
                    "might have to go elsewhere",
                    "looking around for better deals",
                    "shopping around for alternatives",
                    "all your competitors beat your prices",
                    "your competition offers more value",
                    "I found better service elsewhere"
                ],
                "base_risk": 30.0
            },
            "service_frustration": {
                "examples": [
                    # Strong emotional expressions
                    "I'm extremely frustrated with this service",
                    "this is completely unacceptable to me",
                    "I'm really disappointed in your company",
                    "this service is absolutely terrible",
                    "I'm fed up with these constant issues",
                    "this is the worst customer experience",
                    
                    # Moderate frustration
                    "I'm not happy with how this is being handled",
                    "this situation is very disappointing",
                    "I expected much better service than this",
                    "this doesn't meet my expectations at all",
                    "I'm unsatisfied with the quality",
                    "this experience has been quite poor",
                    
                    # Casual/slang expressions
                    "this is BS",
                    "this sucks",
                    "what a joke",
                    "this is ridiculous",
                    "are you kidding me",
                    "this is nonsense",
                    
                    # Specific service issues
                    "your service keeps failing me",
                    "nothing ever works properly",
                    "constant problems with connectivity",
                    "always having technical difficulties",
                    "service outages happen too frequently",
                    "reliability is a major concern"
                ],
                "base_risk": 20.0
            },
            "process_frustration": {
                "examples": [
                    # Repetitive issues - formal
                    "I have to contact you repeatedly about the same issue",
                    "this problem occurs on a recurring basis",
                    "I'm constantly dealing with this same situation",
                    "this keeps happening despite previous resolutions",
                    "I've had to call multiple times about this",
                    "this issue persists even after your fixes",
                    
                    # Repetitive issues - casual
                    "I keep having to call about this",
                    "this happens every single month",
                    "same problem over and over",
                    "here we go again with this",
                    "why does this keep happening",
                    "I'm tired of dealing with this repeatedly",
                    
                    # Time-related frustration
                    "I waste so much time on this",
                    "this takes forever to resolve",
                    "I shouldn't have to keep calling",
                    "too much time spent on simple issues",
                    "constant back and forth is exhausting",
                    "this process is way too complicated",
                    
                    # Systemic issues
                    "your system never works right",
                    "there's always something wrong",
                    "one problem after another",
                    "nothing is ever straightforward",
                    "every interaction is a hassle",
                    "your processes are broken"
                ],
                "base_risk": 20.0
            },
            "positive_resolution": {
                "examples": [
                    # Gratitude - formal
                    "thank you for resolving this matter",
                    "I appreciate your assistance with this issue",
                    "your help has been very valuable",
                    "I'm grateful for your professional service",
                    "thank you for taking care of this promptly",
                    "your support team has been excellent",
                    "No, that's all. Thanks for the help",
                    "Alright, let's go with that then."
                    
                    # Satisfaction - casual
                    "that works for me",
                    "sounds good to me",
                    "I'm happy with that solution",
                    "that's exactly what I needed",
                    "perfect, that fixes everything",
                    "great, I'm satisfied now",
                    
                    # Relief expressions
                    "finally got this sorted out",
                    "glad we could figure this out",
                    "much better now",
                    "that's a relief",
                    "good to have this resolved",
                    "happy this is taken care of",
                    
                    # Acceptance
                    "I can live with that",
                    "that's reasonable",
                    "fair enough",
                    "I'll accept that offer",
                    "that seems like a good compromise",
                    "I'm okay with this arrangement",
                    
                    
                ],
                "base_risk": -40.0
            }
        }
        
        # Pre-compute embeddings for risk patterns
        print("Computing embeddings for risk patterns...")
        self.pattern_embeddings = {}
        for pattern_name, pattern_data in self.risk_patterns.items():
            examples = pattern_data["examples"]
            embeddings = self.embedding_model.encode(examples)
            self.pattern_embeddings[pattern_name] = {
                "embeddings": embeddings,
                "base_risk": pattern_data["base_risk"]
            }
        print("Embeddings computed successfully!")
        
        # Emotion classification API setup
        self.emotion_api_url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
        
        # API URLs for external services
        self.mistral_api_url = "http://localhost:11434/api/generate"  # Ollama endpoint
        
        # Initialize emotion analyzer
        try:
            print("Loading emotion classification model via Hugging Face API...")
            if TRANSFORMERS_AVAILABLE:
                # We'll store the API call logic instead of a local pipeline
                HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                
                # Define a small wrapper function
                def hf_emotion_analyzer(text: str):
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    response = requests.post(self.emotion_api_url, headers=headers, json={"inputs": text})
                    results = response.json()
                    # HF API returns nested list, handle that
                    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                        results = results[0]
                    return results

                self.emotion_analyzer = hf_emotion_analyzer
                print("Emotion classification API is ready!")
            else:
                print("Transformers not available, using basic emotion detection.")
                self.emotion_analyzer = None
        except Exception as e:
            print(f"Could not initialize emotion API, using basic detection: {e}")
            self.emotion_analyzer = None

    def analyze_sentiment(self, text: str, agent_context: str = "") -> Tuple[float, float]:
        """
        Analyze sentiment using both VADER and advanced models if available
        Agent context is stored for future use but not used in sentiment analysis currently
        Returns: (sentiment_score, confidence) where sentiment_score is -1 to 1
        """
        try:
            # Only analyze customer text for sentiment (agent context commented out for now)
            # context_text = f"{agent_context} {text}" if agent_context else text
            customer_text = text
            
            # Use VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(customer_text)
            vader_sentiment = vader_scores['compound']  # Already -1 to 1
            vader_confidence = abs(vader_sentiment)
            
            # Try advanced sentiment analysis if available
            if self.advanced_sentiment:
                try:
                    advanced_results = self.advanced_sentiment(customer_text)[0]
                    
                    # Convert to -1 to 1 scale
                    # Find negative, neutral, positive scores
                    neg_score = next((item['score'] for item in advanced_results if item['label'] == 'LABEL_0'), 0)
                    neu_score = next((item['score'] for item in advanced_results if item['label'] == 'LABEL_1'), 0)
                    pos_score = next((item['score'] for item in advanced_results if item['label'] == 'LABEL_2'), 0)
                    
                    # Convert to compound score similar to VADER
                    advanced_sentiment = pos_score - neg_score
                    advanced_confidence = max(pos_score, neg_score, neu_score)
                    
                    # Combine VADER and advanced model (weighted average)
                    final_sentiment = 0.5 * vader_sentiment + 0.5 * advanced_sentiment
                    final_confidence = 0.5 * vader_confidence + 0.5 * advanced_confidence
                    
                    
                    return final_sentiment, final_confidence
                    
                except Exception as e:
                    print(f"Advanced sentiment analysis failed, using VADER: {e}")
            
            # Fallback to VADER only
            return vader_sentiment, vader_confidence
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0, 0.0

    def analyze_emotion(self, text: str) -> dict:
        """
        Analyze emotion using emotion classification model or basic detection
        Returns: dict with emotion scores
        """
        if self.emotion_analyzer:
            try:
                results = self.emotion_analyzer(text)
            
            # Convert to dictionary of emotion scores
                emotions = {item['label'].lower(): item['score'] for item in results}
                
                
                # Get the dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion[0],
                    'dominant_score': dominant_emotion[1]
                }
                    
            except Exception as e:
                print(f"Emotion analysis error: {e}")
        
        # Fallback to basic emotion detection based on keywords
        return self._basic_emotion_detection(text)
    
    def _basic_emotion_detection(self, text: str) -> dict:
        """Basic emotion detection using keyword matching"""
        text_lower = text.lower()
        
        emotion_keywords = {
            'anger': ['angry', 'furious', 'mad', 'outraged', 'irritated', 'annoyed'],
            'frustration': ['frustrated', 'annoying', 'ridiculous', 'unacceptable', 'tired'],
            'confusion': ['confused', 'unclear', 'understand', 'explain', 'what', 'why', 'how'],
            'relief': ['relief', 'relieved', 'finally', 'good', 'better', 'sorted'],
            'satisfaction': ['satisfied', 'happy', 'great', 'excellent', 'perfect', 'thanks'],
            'fear': ['worried', 'concerned', 'afraid', 'scared', 'anxious'],
            'disappointment': ['disappointed', 'let down', 'expected', 'hoped']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            emotion_scores[emotion] = min(score * 3, 1.0)  # Scale up and cap at 1.0
        
        if emotion_scores and max(emotion_scores.values()) > 0:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return {
                'emotions': emotion_scores,
                'dominant_emotion': dominant_emotion[0],
                'dominant_score': dominant_emotion[1]
            }
        else:
            return {
                'emotions': emotion_scores,
                'dominant_emotion': 'neutral',
                'dominant_score': 0.5
            }

    def detect_risk_patterns_embedding(self, text: str, agent_context: str = "") -> tuple:
        """Detect risk patterns using embedding similarity - only on customer text"""
        # Only analyze customer text for patterns (agent context commented out for now)
        # context_text = f"{agent_context} {text}" if agent_context else text
        customer_text = text
        
        # Get embedding for the customer text only
        text_embedding = self.embedding_model.encode([customer_text])[0]
        
        total_risk_delta = 0.0
        similarity_threshold = 0.45 # Minimum similarity to consider a match
        detected_patterns = []
        
        print("Pattern Analysis:")
        
        # Store all patterns with their similarities
        pattern_similarities = []
        matches_found = False
        
        for pattern_name, pattern_data in self.pattern_embeddings.items():
            # Calculate similarity with all examples in this pattern
            pattern_embeddings = pattern_data["embeddings"]
            similarities = np.dot(pattern_embeddings, text_embedding) / (
                np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            
            # Get the highest similarity for this pattern
            max_similarity = np.max(similarities)
            base_risk = pattern_data["base_risk"]
            
            pattern_similarities.append((pattern_name, max_similarity, base_risk))
            
            if max_similarity > similarity_threshold:
                # Scale risk by similarity (higher similarity = more confident match)
                similarity_factor = (max_similarity - similarity_threshold) / (1.0 - similarity_threshold)
                risk_contribution = base_risk * similarity_factor
                # risk_contribution = base_risk
                total_risk_delta += risk_contribution
                detected_patterns.append(pattern_name)
                print(f"  {pattern_name}: similarity={max_similarity:.3f} (MATCH) -> risk={risk_contribution:.1f}")
                matches_found = True
        
        # If no patterns matched, show only the highest similarity pattern
        if not matches_found:
            best_pattern = max(pattern_similarities, key=lambda x: x[1])
            pattern_name, max_similarity, base_risk = best_pattern
            print(f"  {pattern_name}: similarity={max_similarity:.3f} (highest, below threshold) -> risk=0.0")
        
        return total_risk_delta, detected_patterns

    def detect_risk_patterns_llm(self, text: str, agent_context: str = "") -> tuple:
        """
        Detect risk patterns using LLM indicator extractor
        Returns: (total_risk_delta, detected_patterns)
        """
        print("🤖 LLM Risk Pattern Detection:")
        
        if not self.llm_extractor:
            print("❌ LLM extractor not available, falling back to embedding detection")
            return self.detect_risk_patterns_embedding(text, agent_context)
        
        try:
            # Get LLM analysis
            analysis = self.llm_extractor.get_comprehensive_analysis(text, agent_context)
            
            if "error" in analysis:
                print(f"❌ LLM analysis failed: {analysis['error']}")
                print("⚠️ Falling back to embedding detection")
                return self.detect_risk_patterns_embedding(text, agent_context)
            
            # Map LLM detected patterns to base risk scores (same as embedding system)
            pattern_risk_map = {
                "billing_complaint": 15.0,
                "competitor_mention": 30.0, 
                "service_frustration": 20.0,
                "process_frustration": 20.0,
                "positive_resolution": -40.0
            }
            
            detected_patterns = analysis.get("risk_patterns", [])
            total_risk_delta = 0.0
            
            print(f"  LLM Detected patterns: {detected_patterns}")
            print(f"  LLM Sentiment: {analysis.get('sentiment', 'unknown')}")
            print(f"  LLM Emotion: {analysis.get('emotion', 'unknown')}")
            
            # Calculate risk contribution from detected patterns
            for pattern in detected_patterns:
                if pattern in pattern_risk_map:
                    risk_contribution = pattern_risk_map[pattern]
                    total_risk_delta += risk_contribution
                    print(f"  {pattern}: LLM detected -> risk={risk_contribution:.1f}")
                else:
                    print(f"  {pattern}: unknown pattern, skipping")
            
            if not detected_patterns:
                print("  No risk patterns detected by LLM")
            
            return total_risk_delta, detected_patterns
            
        except Exception as e:
            print(f"❌ Error in LLM pattern detection: {e}")
            print("⚠️ Falling back to embedding detection")
            return self.detect_risk_patterns_embedding(text, agent_context)

    def _convert_llm_sentiment_to_score(self, llm_sentiment: str) -> float:
        """Convert LLM sentiment string to numerical score for scoring logic compatibility"""
        sentiment_mapping = {
            "very_positive": 0.5,
            "positive": 0.3,
            "neutral": 0.0,
            "negative": -0.2,
            "very_negative": -0.7
        }
        return sentiment_mapping.get(llm_sentiment.lower(), 0.0)
    
    def _calculate_llm_pattern_risk(self, detected_patterns: List[str]) -> float:
        """Calculate total pattern risk from LLM detected patterns using same base scores"""
        pattern_risk_map = {
            "billing_complaint": 15.0,
            "competitor_mention": 30.0, 
            "service_frustration": 20.0,
            "process_frustration": 20.0,
            "positive_resolution": -40.0
        }
        
        total_risk = 0.0
        for pattern in detected_patterns:
            if pattern in pattern_risk_map:
                risk_contribution = pattern_risk_map[pattern]
                total_risk += risk_contribution
                print(f"  {pattern}: LLM detected -> risk={risk_contribution:.1f}")
            else:
                print(f"  {pattern}: unknown pattern, skipping")
        
        return total_risk
    
    def _process_with_rule_based_analysis(self, customer_text: str, agent_context: str, timestamp: datetime) -> ChurnEvent:
        """Fallback method to process with rule-based analysis when LLM fails"""
        print("🧠 Using rule-based analysis (fallback)")
        
        # Analyze sentiment with agent context
        sentiment_score, sentiment_confidence = self.analyze_sentiment(customer_text, agent_context)
        
        # Analyze emotion
        emotion_result = self.analyze_emotion(customer_text)
        
        # Detect risk patterns
        pattern_risk, detected_patterns = self.detect_risk_patterns_embedding(customer_text, agent_context)
        
        print(f"Rule-based Sentiment: score={sentiment_score:.3f}, confidence={sentiment_confidence:.3f}")
        print(f"Rule-based Emotion: {emotion_result['dominant_emotion']} ({emotion_result['dominant_score']:.3f})")
        
        # Update conversation context
        self.update_conversation_context(customer_text, detected_patterns)
        
        # Calculate sentiment risk
        sentiment_risk = 0.0
        if sentiment_score < -0.6:  # Very negative
            sentiment_risk = 30.0
        elif sentiment_score < -0.3:  # Moderately negative
            sentiment_risk = 20.0
        elif sentiment_score < -0.1:  # Slightly negative
            sentiment_risk = 10.0
        elif sentiment_score > 0.36:  # Positive
            sentiment_risk = -20.0
        elif sentiment_score > 0.25:  # Slightly positive
            sentiment_risk = -10.0
        
        print(f"Sentiment Risk: {sentiment_risk:.1f} (based on sentiment score {sentiment_score:.3f})")
        
        # Calculate emotion risk
        print("Emotion Risk Analysis:")
        emotion_risk = self.calculate_emotion_risk(emotion_result)
        print(f"Emotion Risk: {emotion_risk:.1f}")
        
        total_risk_delta = pattern_risk + sentiment_risk + emotion_risk
        print(f"Total Risk Delta: {pattern_risk:.1f} + {sentiment_risk:.1f} + {emotion_risk:.1f} = {total_risk_delta:.1f}")
        
        # Update cumulative score
        previous_score = self.current_score
        alpha = self.alpha
        new_score = (1 - alpha) * previous_score + alpha * (previous_score + total_risk_delta)
        new_score = max(0.0, min(100.0, new_score))
        
        print(f"Score Update: ({1-alpha:.1f} × {previous_score:.1f}) + ({alpha:.1f} × ({previous_score:.1f} + {total_risk_delta:.1f})) = {new_score:.1f}")
        
        self.current_score = new_score
        
        # Create event record
        event = ChurnEvent(
            timestamp=timestamp,
            speaker="Customer",
            text=customer_text,
            agent_context=agent_context,
            sentiment_score=sentiment_score,
            emotion_result=emotion_result,
            risk_delta=total_risk_delta,
            cumulative_score=new_score,
            confidence=sentiment_confidence,
            detected_patterns=detected_patterns
        )
        
        # Store in history
        self.conversation_history.append(event)
        self.risk_events.append(event)
        
        return event

    def calculate_emotion_risk(self, emotion_result: dict) -> float:
        """Calculate risk based on dominant emotion only"""
        dominant_emotion = emotion_result.get('dominant_emotion', 'neutral')
        dominant_score = emotion_result.get('dominant_score', 0.0)
        
        # Define risk values for each emotion
        emotion_risk_mapping = {
            'anger': 25.0,      # High risk - angry customers likely to churn
            'disgust': 20.0,    # High risk - disgusted with service
            'fear': 10.0,        # Moderate risk - worried about service
            'joy': -30.0,       # Negative risk - happy customers
            'neutral': 0.0,     # No risk impact
            'sadness': 10.0,    # Moderate risk - disappointed customers
            'surprise': 5.0,   # Low risk - unexpected situations
            'frustration': 10.0 # Moderate risk - frustrated customers
        }
        
        # Calculate risk only for the dominant emotion
        emotion_risk = 0.0
        if dominant_emotion in emotion_risk_mapping:
            emotion_risk = emotion_risk_mapping[dominant_emotion]
            print(f"  {dominant_emotion}: score={dominant_score:.3f} -> risk={emotion_risk:.1f}")
        else:
            print(f"  {dominant_emotion}: score={dominant_score:.3f} -> risk=0.0 (unknown emotion)")
        
        return emotion_risk

    def extract_prices_from_text(self, text: str) -> List[float]:
        """Extract mentioned prices from customer text using regex patterns"""
        
        # Dictionary for converting written numbers to digits
        word_to_number = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }
        
        def words_to_number(text_phrase):
            """Convert written numbers to digits"""
            text_phrase = text_phrase.lower().strip()
            
            # Handle special cases like "two hundred", "one hundred and eighty"
            if 'hundred' in text_phrase:
                parts = text_phrase.split('hundred')
                if len(parts) == 2:
                    # Get the number before "hundred"
                    hundreds_part = parts[0].strip()
                    remainder_part = parts[1].strip()
                    
                    # Convert hundreds
                    hundreds_value = 0
                    if hundreds_part:
                        if hundreds_part in word_to_number:
                            hundreds_value = word_to_number[hundreds_part] * 100
                        else:
                            return None
                    else:
                        hundreds_value = 100  # Just "hundred"
                    
                    # Convert remainder (after "and" if present)
                    remainder_value = 0
                    if remainder_part:
                        remainder_part = remainder_part.replace('and', '').strip()
                        if remainder_part in word_to_number:
                            remainder_value = word_to_number[remainder_part]
                        else:
                            # Handle compound numbers like "eighty dollars"
                            remainder_words = remainder_part.split()
                            for word in remainder_words:
                                if word in word_to_number:
                                    remainder_value += word_to_number[word]
                    
                    return hundreds_value + remainder_value
            
            # Handle simple word numbers
            if text_phrase in word_to_number:
                return word_to_number[text_phrase]
            
            return None
        
        # Original numeric patterns
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $200, $180.50
            r'(\d+) dollars?',       # 200 dollars
            r'paying (\d+)',         # paying 200
            r'bill (?:is|was) (\d+)', # bill is 200
            r'costs? (\d+)',         # costs 200
            r'charge[ds]? (\d+)'     # charged 200
        ]
        
        # Written number patterns
        written_patterns = [
            r"it\'s\s+([\w\s]+?)\s+dollars?",  # it's two hundred dollars
            r"was\s+([\w\s]+?)\s+dollars?",   # was one hundred and eighty dollars  
            r"bill\s+(?:is|was)\s+([\w\s]+?)\s+dollars?",  # bill is two hundred dollars
            r"paying\s+([\w\s]+?)\s+dollars?",  # paying one hundred fifty dollars
            r"(?:cost|costs?)\s+(?:is|are)\s+([\w\s]+?)\s+dollars?",  # cost is three hundred dollars
            r"((?:\w+\s+)*(?:hundred|thousand)(?:\s+and\s+\w+)*)\s+dollars?",  # general pattern for written numbers
        ]
        
        prices = []
        text_lower = text.lower()
        
        # Extract numeric prices
        for pattern in price_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    price = float(match)
                    if 10 <= price <= 1000:  # Reasonable price range for telecom bills
                        prices.append(price)
                except ValueError:
                    continue
        
        # Extract written number prices
        for pattern in written_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    price = words_to_number(match)
                    if price is not None and 10 <= price <= 1000:
                        prices.append(float(price))
                except:
                    continue
        
        return prices

    def detect_unused_services(self, text: str) -> List[str]:
        """Detect services that customer mentions as unused or underutilized"""
        text_lower = text.lower()
        unused_services = []
        
        # Service usage patterns
        tv_unused_patterns = [
            'barely watch tv', 'don\'t watch tv', 'never use tv', 'rarely watch',
            'don\'t need tv', 'hardly watch', 'not watching tv', 'tv unused'
        ]
        
        mobile_unused_patterns = [
            'don\'t use mobile', 'rarely call', 'never use phone', 'mobile unused',
            'don\'t need mobile', 'hardly call'
        ]
        
        internet_unused_patterns = [
            'slow internet', 'don\'t need fast', 'basic internet', 'light usage'
        ]
        
        # Check for TV service issues
        for pattern in tv_unused_patterns:
            if pattern in text_lower:
                unused_services.append('TV')
                break
        
        # Check for Mobile service issues  
        for pattern in mobile_unused_patterns:
            if pattern in text_lower:
                unused_services.append('Mobile')
                break
                
        # Check for Internet service issues
        for pattern in internet_unused_patterns:
            if pattern in text_lower:
                unused_services.append('Internet')
                break
        
        return list(set(unused_services))  # Remove duplicates

    def detect_competitor_mentions(self, text: str) -> List[str]:
        """Detect competitor mentions in customer text"""
        text_lower = text.lower()
        competitors = ['tmobile', 't-mobile', 'verizon', 'att', 'at&t', 'comcast', 
                      'spectrum', 'xfinity', 'cox', 'charter', 'directv', 'dish']
        
        mentioned_competitors = []
        for competitor in competitors:
            if competitor in text_lower:
                mentioned_competitors.append(competitor.upper())
        
        return mentioned_competitors

    def update_conversation_context(self, customer_text: str, detected_patterns: List[str]):
        """Update conversation context with new information from customer message"""
        
        # Extract prices
        new_prices = self.extract_prices_from_text(customer_text)
        self.conversation_context.mentioned_prices.extend(new_prices)
        
        # Update current bill and price ceiling
        if new_prices:
            self.conversation_context.current_bill = max(new_prices)
            self.conversation_context.price_ceiling = max(new_prices)
        
        # Detect unused services
        unused = self.detect_unused_services(customer_text)
        for service in unused:
            if service not in self.conversation_context.unused_services:
                self.conversation_context.unused_services.append(service)
        
        # Detect competitor mentions
        competitors = self.detect_competitor_mentions(customer_text)
        for competitor in competitors:
            if competitor not in self.conversation_context.competitor_mentions:
                self.conversation_context.competitor_mentions.append(competitor)
        
        # Update patterns detected
        self.conversation_context.patterns_detected = detected_patterns
        
        # Determine price trend (simple logic)
        if len(self.conversation_context.mentioned_prices) >= 2:
            recent_prices = self.conversation_context.mentioned_prices[-2:]
            if recent_prices[1] > recent_prices[0]:
                self.conversation_context.price_trend = 'increasing'
            elif recent_prices[1] < recent_prices[0]:
                self.conversation_context.price_trend = 'decreasing'
            else:
                self.conversation_context.price_trend = 'stable'

    def process_customer_message(self, customer_text: str, agent_context: str = "") -> ChurnEvent:
        """Process a customer message and update churn score (only for customer messages)"""
        timestamp = datetime.now()
        
        print("--- Churn Analysis ---")
        
        if self.use_llm_indicators and self.llm_extractor:
            # Use LLM for all analysis: sentiment, emotion, and risk patterns
            print("🤖 Using LLM-based analysis for sentiment, emotion, and risk patterns")
            
            try:
                # Get comprehensive LLM analysis
                llm_analysis = self.llm_extractor.get_comprehensive_analysis(customer_text, agent_context)
                
                if "error" in llm_analysis:
                    print(f"❌ LLM analysis failed: {llm_analysis['error']}")
                    print("⚠️ Falling back to rule-based analysis")
                    return self._process_with_rule_based_analysis(customer_text, agent_context, timestamp)
                
                # Extract LLM results
                llm_sentiment = llm_analysis.get("sentiment", "neutral")
                llm_emotion = llm_analysis.get("emotion", "neutral")
                detected_patterns = llm_analysis.get("risk_patterns", [])
                
                print(f"LLM Sentiment: {llm_sentiment}")
                print(f"LLM Emotion: {llm_emotion}")
                print(f"LLM Risk Patterns: {detected_patterns}")
                
                # Convert LLM sentiment to numerical score (for scoring logic compatibility)
                sentiment_score = self._convert_llm_sentiment_to_score(llm_sentiment)
                sentiment_confidence = 0.8  # High confidence for LLM
                
                # Create emotion result in expected format
                emotion_result = {
                    'dominant_emotion': llm_emotion,
                    'dominant_score': 0.8  # High confidence for LLM
                }
                
                # Calculate pattern risk using same base scores
                pattern_risk = self._calculate_llm_pattern_risk(detected_patterns)
                
                print(f"Converted sentiment score: {sentiment_score:.3f}")
                print(f"Pattern risk from LLM: {pattern_risk:.1f}")
                
            except Exception as e:
                print(f"❌ Error in LLM analysis: {e}")
                print("⚠️ Falling back to rule-based analysis")
                return self._process_with_rule_based_analysis(customer_text, agent_context, timestamp)
        
        else:
            # Use rule-based analysis
            print("🧠 Using rule-based analysis for sentiment, emotion, and risk patterns")
            sentiment_score, sentiment_confidence = self.analyze_sentiment(customer_text, agent_context)
            emotion_result = self.analyze_emotion(customer_text)
            pattern_risk, detected_patterns = self.detect_risk_patterns_embedding(customer_text, agent_context)
            
            print(f"Rule-based Sentiment: score={sentiment_score:.3f}, confidence={sentiment_confidence:.3f}")
            print(f"Rule-based Emotion: {emotion_result['dominant_emotion']} ({emotion_result['dominant_score']:.3f})")
        
        # Update conversation context
        self.update_conversation_context(customer_text, detected_patterns)
        
        # Calculate sentiment risk (same logic for both LLM and rule-based)
        sentiment_risk = 0.0
        if sentiment_score < -0.6:  # Very negative
            sentiment_risk = 30.0
        elif sentiment_score < -0.3:  # Moderately negative
            sentiment_risk = 20.0
        elif sentiment_score < -0.1:  # Slightly negative
            sentiment_risk = 10.0
        elif sentiment_score > 0.36:  # Positive
            sentiment_risk = -20.0
        elif sentiment_score > 0.25:  # Slightly positive
            sentiment_risk = -10.0
        
        print(f"Sentiment Risk: {sentiment_risk:.1f} (based on sentiment score {sentiment_score:.3f})")
        
        # Calculate emotion risk (same logic for both LLM and rule-based)
        print("Emotion Risk Analysis:")
        emotion_risk = self.calculate_emotion_risk(emotion_result)
        print(f"Emotion Risk: {emotion_risk:.1f}")
        
        total_risk_delta = pattern_risk + sentiment_risk + emotion_risk
        print(f"Total Risk Delta: {pattern_risk:.1f} + {sentiment_risk:.1f} + {emotion_risk:.1f} = {total_risk_delta:.1f}")
        
        # Update cumulative score with detailed calculation (same logic for both)
        previous_score = self.current_score
        alpha = self.alpha
        new_score = (1 - alpha) * previous_score + alpha * (previous_score + total_risk_delta)
        new_score = max(0.0, min(100.0, new_score))
        
        print(f"Score Update: ({1-alpha:.1f} × {previous_score:.1f}) + ({alpha:.1f} × ({previous_score:.1f} + {total_risk_delta:.1f})) = {new_score:.1f}")
        
        self.current_score = new_score
        
        # Create event record
        event = ChurnEvent(
            timestamp=timestamp,
            speaker="Customer",
            text=customer_text,
            agent_context=agent_context,
            sentiment_score=sentiment_score,
            emotion_result=emotion_result,
            risk_delta=total_risk_delta,
            cumulative_score=new_score,
            confidence=sentiment_confidence,
            detected_patterns=detected_patterns
        )
        
        # Store in history
        self.conversation_history.append(event)
        self.risk_events.append(event)
        
        return event

    def apply_rule_based_filtering(self, offers: List[Dict]) -> List[Dict]:
        """Apply rule-based filtering with rejection reasons for all offers"""
        
        print(f"\n🔧 RULE-BASED FILTERING:")
        print(f"Starting with {len(offers)} offers")
        
        processed_offers = []
        
        for offer in offers:
            offer_copy = offer.copy()
            offer_copy['accepted'] = True
            offer_copy['rejection_reason'] = None
            
            # Check price ceiling
            if self.conversation_context.price_ceiling and offer['price_delta'] > self.conversation_context.price_ceiling:
                offer_copy['accepted'] = False
                offer_copy['rejection_reason'] = f"Price ${offer['price_delta']} exceeds budget ${self.conversation_context.price_ceiling}"
            
            # Check for unused services
            elif self.conversation_context.unused_services:
                offer_products = set(offer.get('product_types', []))
                unused_services = set(self.conversation_context.unused_services)
                if offer_products & unused_services:  # If offer includes unused services
                    overlap = offer_products & unused_services
                    offer_copy['accepted'] = False
                    offer_copy['rejection_reason'] = f"Includes less used service: {', '.join(overlap)}"
            
            processed_offers.append(offer_copy)
        
        # Sort offers: accepted first, then rejected
        accepted_offers = [o for o in processed_offers if o['accepted']]
        rejected_offers = [o for o in processed_offers if not o['accepted']]
        
        # Apply sorting to accepted offers based on context
        if self.conversation_context.competitor_mentions:
            print("Competitor mentions detected - sorting by price")
            accepted_offers.sort(key=lambda x: x['price_delta'])
        elif self.conversation_context.mentioned_prices or 'billing_complaint' in self.conversation_context.patterns_detected:
            print("Price concerns detected - sorting by price")
            accepted_offers.sort(key=lambda x: x['price_delta'])
        else:
            print("Default sorting - by features and relevance")
            accepted_offers.sort(key=lambda x: (-x.get('relevance_score', 0), len(x.get('product_types', []))))
        
        # Combine: accepted offers first, then rejected at the end
        final_offers = accepted_offers + rejected_offers
        
        accepted_count = len(accepted_offers)
        rejected_count = len(rejected_offers)
        
        print(f"Final result: {accepted_count} accepted, {rejected_count} rejected offers")
        
        return final_offers

    def apply_llm_based_filtering(self, offers: List[Dict], customer_text: str) -> List[Dict]:
        """Apply LLM-based filtering using comprehensive analysis from LLM extractor"""
        
        print(f"\n🤖 LLM-BASED FILTERING:")
        print(f"Starting with {len(offers)} offers")
        
        if not self.llm_extractor:
            print("❌ LLM extractor not available, falling back to rule-based filtering")
            return self.apply_rule_based_filtering(offers)
        
        try:
            # Get comprehensive LLM analysis
            llm_analysis = self.llm_extractor.get_comprehensive_analysis(customer_text, "")
            
            if "error" in llm_analysis:
                print(f"❌ LLM analysis failed: {llm_analysis['error']}")
                print("⚠️ Falling back to rule-based filtering")
                return self.apply_rule_based_filtering(offers)
            
            # Use LLM extractor's sophisticated filtering logic
            filtered_offers = llm_analysis.get("filtered_offers", [])
            
            if not filtered_offers:
                print("⚠️ No filtered offers from LLM, falling back to rule-based filtering")
                return self.apply_rule_based_filtering(offers)
            
            # Convert LLM filtered offers back to the format expected by simple churn scorer
            processed_offers = []
            
            # Create a mapping of offer IDs to original offers
            offer_map = {offer['offer_id']: offer for offer in offers}
            
            # Process LLM filtered offers
            for llm_offer in filtered_offers:
                offer_id = llm_offer.get('offer_id')
                if offer_id in offer_map:
                    original_offer = offer_map[offer_id].copy()
                    
                    # Add LLM filtering information
                    original_offer['accepted'] = True  # LLM filtered offers are considered accepted
                    original_offer['rejection_reason'] = None
                    original_offer['llm_filtered'] = True
                    
                    # Add LLM filtering explanations if available
                    if 'filtering_explanation' in llm_offer:
                        original_offer['llm_explanation'] = llm_offer['filtering_explanation']
                    
                    # Add relevance information from LLM
                    if 'relevance_score' in llm_offer:
                        original_offer['llm_relevance'] = llm_offer['relevance_score']
                    
                    processed_offers.append(original_offer)
            
            # Add rejected offers (those not in LLM filtered list)
            llm_offer_ids = {offer.get('offer_id') for offer in filtered_offers}
            for offer in offers:
                if offer['offer_id'] not in llm_offer_ids:
                    rejected_offer = offer.copy()
                    rejected_offer['accepted'] = False
                    rejected_offer['rejection_reason'] = "Filtered out by LLM analysis"
                    rejected_offer['llm_filtered'] = False
                    processed_offers.append(rejected_offer)
            
            # Sort: accepted (LLM filtered) first, then rejected
            accepted_offers = [o for o in processed_offers if o['accepted']]
            rejected_offers = [o for o in processed_offers if not o['accepted']]
            
            # LLM already provides intelligent sorting, so we keep the order
            final_offers = accepted_offers + rejected_offers
            
            accepted_count = len(accepted_offers)
            rejected_count = len(rejected_offers)
            
            print(f"LLM Analysis:")
            print(f"  Sentiment: {llm_analysis.get('sentiment', 'unknown')}")
            print(f"  Emotion: {llm_analysis.get('emotion', 'unknown')}")
            print(f"  Risk Patterns: {llm_analysis.get('risk_patterns', [])}")
            print(f"Final result: {accepted_count} accepted, {rejected_count} rejected offers")
            
            return final_offers
            
        except Exception as e:
            print(f"❌ Error in LLM offer filtering: {e}")
            print("⚠️ Falling back to rule-based filtering")
            return self.apply_rule_based_filtering(offers)

    def get_dynamic_offers(self, customer_text: str, max_offers: int = 3) -> Optional[Dict]:
        """Get dynamic offer recommendations based on current conversation and churn score"""
        if not self.offer_engine:
            print("Offer engine not available")
            return None
        
        try:
            # Calculate churn delta from last event
            churn_delta = 0.0
            if len(self.risk_events) >= 2:
                churn_delta = self.risk_events[-1].cumulative_score - self.risk_events[-2].cumulative_score
            elif len(self.risk_events) == 1:
                churn_delta = self.risk_events[-1].cumulative_score - self.baseline_score
            
            # Get offer recommendations
            recommendations = self.offer_engine.get_recommended_offers(
                customer_text=customer_text,
                churn_score=self.current_score,
                churn_delta=churn_delta,
                previous_offers=self.previous_offers,
                max_offers=max_offers
            )
            
            # Track shown offers
            if recommendations and 'offers' in recommendations:
                new_offer_ids = [offer['offer_id'] for offer in recommendations['offers']]
                self.previous_offers.extend(new_offer_ids)
                # Keep only last 10 offers to avoid memory bloat
                if len(self.previous_offers) > 10:
                    self.previous_offers = self.previous_offers[-10:]
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting dynamic offers: {e}")
            return None

    def get_offers_for_agent(self, customer_text: str) -> List[Dict]:
        """Get simplified offer recommendations for agent display"""
        
        # Choose filtering method based on flags
        if self.use_llm_offer_filtering and self.llm_extractor:
            filtered_offers = self.apply_llm_based_filtering(self.initial_offers, customer_text)
        else:
            # For rule-based filtering, work directly with static catalog to get all offers
            # Apply rule-based filtering to static offers
            filtered_offers = self.apply_rule_based_filtering(self.initial_offers)
        
        # Convert to simplified format for frontend
        simplified_offers = []
        for offer in filtered_offers:
            simplified_offers.append({
                'id': offer['offer_id'],
                'title': offer['title'], 
                'description': offer['description'],
                'value': f"${offer['price_delta']}/month",
                'urgency': offer.get('urgency', 'Standard'),
                'category': offer['category'],
                'relevance': int(offer.get('relevance_score', 80)),
                'type': offer['category'],
                'price_delta': offer['price_delta'],
                'retention_offer': offer['retention_offer'],
                'accepted': offer.get('accepted', True),
                'rejection_reason': offer.get('rejection_reason', None)
            })
        
        # Show context information
        if self.conversation_context.mentioned_prices:
            print(f"\n💰 Context: Prices mentioned: {self.conversation_context.mentioned_prices}")
        if self.conversation_context.unused_services:
            print(f"📺 Context: Unused services: {self.conversation_context.unused_services}")
        if self.conversation_context.competitor_mentions:
            print(f"🏃 Context: Competitors mentioned: {self.conversation_context.competitor_mentions}")
        
        return simplified_offers
    
    def generate_conversation_note(self, transcript_messages: List[Dict]) -> str:
        """Generate a short summary of the last 3-4 conversation messages using Ollama Mistral"""
        try:
            # Get last 3-4 messages (customer and agent)
            recent_messages = transcript_messages[-6:] if len(transcript_messages) >= 6 else transcript_messages
            
            # Format conversation for the prompt
            conversation_text = ""
            for msg in recent_messages:
                speaker = "Customer" if msg.get('type') == 'customer' else "Agent"
                conversation_text += f"{speaker}: {msg.get('text', '')}\n"
            
            if not conversation_text.strip():
                return "No recent conversation to summarize"
            
            # Prepare prompt for Ollama Mistral
            prompt = f"Summarize this recent customer service conversation in 1-2 sentences:\n\n{conversation_text}\n\nSummary:"
            
            # Call Ollama API
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.mistral_api_url, json=payload)
            
            # Better error handling
            if response.status_code != 200:
                print(f"Ollama API Error {response.status_code}: {response.text}")
                return f"API Error: {response.status_code}"
            
            try:
                result = response.json()
            except ValueError:
                print(f"Invalid JSON response: {response.text}")
                return "Invalid API response format"
            
            if isinstance(result, dict) and 'response' in result:
                return result['response'].strip()
            else:
                print(f"Unexpected API response format: {result}")
                return "Unexpected response format"
                
        except Exception as e:
            print(f"Error generating conversation note: {e}")
            return f"Error: {str(e)}"
    
    def generate_call_summary(self, transcript_messages: List[Dict]) -> str:
        """Generate a comprehensive call summary using Ollama Mistral"""
        try:
            # Format entire conversation
            conversation_text = ""
            for msg in transcript_messages:
                speaker = "Customer" if msg.get('type') == 'customer' else "Agent"
                conversation_text += f"{speaker}: {msg.get('text', '')}\n"
            
            if not conversation_text.strip():
                return "No conversation to summarize"
            
            # Prepare comprehensive summary prompt for Ollama Mistral
            prompt = f"Provide a comprehensive summary of this customer service call including key issues, solutions discussed, and outcomes:\n\n{conversation_text}\n\nDetailed Summary:"
            
            # Call Ollama API
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.mistral_api_url, json=payload)
            
            # Better error handling
            if response.status_code != 200:
                print(f"Ollama API Error {response.status_code}: {response.text}")
                return f"API Error: {response.status_code}"
            
            try:
                result = response.json()
            except ValueError:
                print(f"Invalid JSON response: {response.text}")
                return "Invalid API response format"
            
            if isinstance(result, dict) and 'response' in result:
                return result['response'].strip()
            else:
                print(f"Unexpected API response format: {result}")
                return "Unexpected response format"
                
        except Exception as e:
            print(f"Error generating call summary: {e}")
            return f"Error: {str(e)}"

    def should_trigger_offer_update(self, churn_delta_threshold: float = 5.0) -> bool:
        """Determine if offers should be updated based on churn score changes"""
        print(f"🔍 should_trigger_offer_update: threshold={churn_delta_threshold}, events={len(self.risk_events)}")
        
        if len(self.risk_events) < 2:
            print("✅ Triggering offer update - fewer than 2 events")
            return True  # Always update on first few events
        
        last_delta = self.risk_events[-1].cumulative_score - self.risk_events[-2].cumulative_score
        print(f"📊 Score change: {self.risk_events[-2].cumulative_score:.1f} → {self.risk_events[-1].cumulative_score:.1f} (delta: {last_delta:.1f})")
        should_trigger = abs(last_delta) >= churn_delta_threshold
        print(f"{'✅' if should_trigger else '❌'} Should trigger: {should_trigger} (|{last_delta:.1f}| >= {churn_delta_threshold})")
        return should_trigger

    def get_current_score(self) -> float:
        """Get current churn score"""
        return round(self.current_score, 1)

    def set_baseline_score(self, score: float):
        """Set the baseline score for the customer"""
        self.baseline_score = max(0.0, min(100.0, score))
        self.current_score = self.baseline_score
        print(f"🎯 Set baseline churn score to {self.baseline_score}/100")

    def set_customer_profile_data(self, customer_data: dict):
        """Set customer profile data for LLM indicator extractor"""
        print(f"📋 Received customer data: {customer_data}")
        
        if self.llm_extractor:
            # Extract MRC from string format like "$200" to integer
            current_mrc = customer_data.get('currentMRC', '200')
            print(f"📋 Raw currentMRC: {current_mrc} (type: {type(current_mrc)})")
            if isinstance(current_mrc, str):
                # Remove $ symbol and convert to int
                current_mrc = int(current_mrc.replace('$', ''))
            
            previous_mrc = customer_data.get('previousMRC', '180')
            print(f"📋 Raw previousMRC: {previous_mrc} (type: {type(previous_mrc)})")
            if isinstance(previous_mrc, str):
                previous_mrc = int(previous_mrc.replace('$', ''))
            
            # Update the LLM extractor's customer profile
            updated_profile = {
                "name": customer_data.get('name', 'Unknown Customer'),
                "current_mrc": current_mrc,
                "previous_mrc": previous_mrc,
                "tenure_months": self._parse_tenure(customer_data.get('tenure', '18 months')),
                "current_plan": customer_data.get('currentPlan', ''),
                "services": customer_data.get('currentProducts', 'Internet, TV, Mobile').split(', ')
            }
            
            # print(f"📋 Before update - LLM extractor profile: {self.llm_extractor.customer_profile}")
            self.llm_extractor.customer_profile.update(updated_profile)
            # print(f"📋 After update - LLM extractor profile: {self.llm_extractor.customer_profile}")
            # print(f"💼 Updated customer profile: {self.llm_extractor.customer_profile['name']} (MRC: ${self.llm_extractor.customer_profile['current_mrc']})")
        else:
            print("⚠️ LLM extractor not available, cannot set customer profile")
    
    def _parse_tenure(self, tenure_str: str) -> int:
        """Parse tenure string like '18 months' to integer months"""
        try:
            return int(tenure_str.split()[0])
        except (ValueError, IndexError):
            return 18  # Default fallback

    def reset_conversation(self):
        """Reset all conversation state including churn score and LLM rejected offers"""
        self.current_score = self.baseline_score  # Reset to customer's baseline instead of hardcoded 50
        self.conversation_history = []
        self.risk_events = []
        
        # Reset conversation context
        self.conversation_context = ConversationContext(
            mentioned_prices=[],
            unused_services=[],
            price_trend='stable',
            patterns_detected=[],
            sentiment_trend='neutral',
            competitor_mentions=[]
        )
        
        # Reset LLM rejected offers
        self.llm_rejected_offers = {}
        print(f"🔄 Reset conversation state to baseline score {self.baseline_score}/100")

    def reset_llm_rejected_offers(self):
        """Reset only the LLM rejected offers (useful for testing or new conversation)"""
        self.llm_rejected_offers = {}
        print("🔄 Reset LLM rejected offers - all offers available again")

    def get_offers_for_agent_with_analysis(self, customer_text: str, llm_analysis: dict = None) -> List[Dict]:
        """
        Get offers using pre-computed LLM analysis to avoid duplicate calls
        In hybrid mode: Always use LLM filtering when analysis is provided
        Always returns ALL initial offers with accept/reject status for frontend display
        """
        if llm_analysis and self.llm_extractor:
            # Always use ALL initial offers for consistent frontend display
            print(f"🔄 Starting LLM filtering with ALL {len(self.initial_offers)} initial offers for frontend")
            
            # Use LLM filtering when analysis is provided (hybrid mode or LLM mode)
            filtered_offers = self.apply_llm_based_filtering_with_analysis(self.initial_offers, llm_analysis)
        else:
            # Fallback to rule-based filtering (should only happen in rule-based mode)
            filtered_offers = self.apply_rule_based_filtering(self.initial_offers)
        
        # Return all offers in frontend format (both accepted and rejected)
        frontend_offers = []
        for offer in filtered_offers:
            frontend_offer = {
                'id': offer['offer_id'],
                'title': offer['title'],
                'description': offer['description'],
                'value': f"${offer['price_delta']}/month" if offer['price_delta'] != 0 else "No cost change",
                'urgency': offer.get('urgency', 'Standard'),
                'category': offer['category'],
                'relevance': offer.get('relevance', 50),
                'type': offer['category'],
                'price_delta': offer['price_delta'],
                'retention_offer': offer.get('retention_offer', False),
                'accepted': offer.get('accepted', True),
                'rejection_reason': offer.get('rejection_reason'),
                'llm_filtered': offer.get('llm_filtered', False)
            }
            frontend_offers.append(frontend_offer)
        
        return frontend_offers

    def apply_llm_based_filtering_with_analysis(self, offers: List[Dict], llm_analysis: dict) -> List[Dict]:
        """
        Apply LLM-based filtering using pre-computed analysis
        In hybrid mode: NO fallback to rule-based filtering, return existing offers if LLM fails
        Maintains persistent rejected offers - once rejected by LLM, stays rejected (greyed out)
        """
        print(f"\n🤖 LLM-BASED FILTERING (using existing analysis):")
        print(f"Starting with {len(offers)} offers")
        
        try:
            if "error" in llm_analysis:
                print(f"❌ LLM analysis contains error: {llm_analysis['error']}")
                print("🔄 Hybrid mode: Returning all offers as accepted (LLM error fallback)")
                # Return all offers as accepted when LLM fails (fallback case)
                return self._format_offers_as_accepted(offers)
            
            # Use the existing analysis instead of calling LLM again
            filtered_offers = llm_analysis.get("filtered_offers", [])
            if not filtered_offers:
                print("⚠️ No filtered offers from LLM analysis")
                print("🔄 Hybrid mode: Returning all offers as rejected (filtering resulted in 0 offers)")
                # Return all offers as rejected since filtering legitimately resulted in 0 offers
                return self._format_offers_as_rejected(offers, llm_analysis)
            
            # ALWAYS work with ALL initial offers for frontend display
            all_initial_offers = self.initial_offers
            print(f"🔄 Processing ALL {len(all_initial_offers)} offers for frontend display")
            
            # Track previously rejected offers
            if hasattr(self, 'llm_rejected_offers'):
                print(f"🔄 Found {len(self.llm_rejected_offers)} previously rejected offers")
            else:
                self.llm_rejected_offers = {}
            
            processed_offers = []
            
            # Get LLM-accepted offer IDs for this call
            llm_offer_ids = {offer.get('offer_id') for offer in filtered_offers}
            
            # Extract indicators for generating rejection reasons
            sentiment = llm_analysis.get('sentiment', 'neutral')
            emotion = llm_analysis.get('emotion', 'neutral')
            risk_patterns = llm_analysis.get('risk_patterns', [])
            offer_indicators = llm_analysis.get('offer_indicators', {})
            
            # Process ALL offers - check both current LLM decision and persistent rejections
            newly_rejected = 0
            for offer in all_initial_offers:
                offer_id = offer['offer_id']
                
                # Check if this offer was previously rejected by LLM (persistent)
                if offer_id in self.llm_rejected_offers:
                    # This offer was rejected in a previous LLM call - keep it rejected
                    rejected_offer = offer.copy()
                    rejected_offer['accepted'] = False
                    rejected_offer['llm_filtered'] = False
                    rejected_offer['rejection_reason'] = self.llm_rejected_offers[offer_id]
                    rejected_offer['persistent_rejection'] = True
                    processed_offers.append(rejected_offer)
                    print(f"  📋 {offer_id}: Previously rejected - {self.llm_rejected_offers[offer_id][:50]}...")
                
                elif offer_id in llm_offer_ids:
                    # This offer is accepted by LLM in this call
                    accepted_offer = offer.copy()
                    accepted_offer['accepted'] = True
                    accepted_offer['rejection_reason'] = None
                    accepted_offer['llm_filtered'] = True
                    
                    # Find corresponding LLM offer for additional details
                    for llm_offer in filtered_offers:
                        if llm_offer.get('offer_id') == offer_id:
                            if 'filtering_explanation' in llm_offer:
                                accepted_offer['llm_explanation'] = llm_offer['filtering_explanation']
                            if 'relevance_score' in llm_offer:
                                accepted_offer['llm_relevance'] = llm_offer['relevance_score']
                            break
                    
                    processed_offers.append(accepted_offer)
                    print(f"  ✅ {offer_id}: Accepted by LLM")
                
                else:
                    # This offer is rejected by LLM in this call - add to persistent rejections
                    rejection_reason = self._generate_enhanced_llm_rejection_reason(
                        offer, sentiment, emotion, risk_patterns, offer_indicators
                    )
                    
                    # Store in persistent rejected offers
                    self.llm_rejected_offers[offer_id] = rejection_reason
                    
                    rejected_offer = offer.copy()
                    rejected_offer['accepted'] = False
                    rejected_offer['llm_filtered'] = False
                    rejected_offer['rejection_reason'] = rejection_reason
                    rejected_offer['persistent_rejection'] = False  # Newly rejected
                    
                    processed_offers.append(rejected_offer)
                    newly_rejected += 1
                    print(f"  ❌ {offer_id}: Newly rejected - {rejection_reason[:50]}...")
            
            accepted_offers = [o for o in processed_offers if o['accepted']]
            rejected_offers = [o for o in processed_offers if not o['accepted']]
            
            # Sort accepted offers first, then rejected
            final_offers = accepted_offers + rejected_offers
            
            accepted_count = len(accepted_offers)
            rejected_count = len(rejected_offers)
            persistent_count = len([o for o in rejected_offers if o.get('persistent_rejection', False)])
            
            print(f"LLM Analysis (using existing):")
            print(f"  Sentiment: {llm_analysis.get('sentiment', 'unknown')}")
            print(f"  Emotion: {llm_analysis.get('emotion', 'unknown')}")
            print(f"  Risk Patterns: {llm_analysis.get('risk_patterns', [])}")
            print(f"Final result: {accepted_count} accepted, {rejected_count} rejected ({persistent_count} persistent, {newly_rejected} new)")
            print(f"🔄 Frontend will display ALL {len(final_offers)} offers with persistent rejection state")
            
            return final_offers
            
        except Exception as e:
            print(f"❌ Error in LLM offer filtering with existing analysis: {e}")
            print("🔄 Hybrid mode: Returning all offers as accepted (exception fallback)")
            # Return all offers as accepted when exception occurs (fallback case)
            return self._format_offers_as_accepted(offers)

    def _format_offers_as_accepted(self, offers: List[Dict]) -> List[Dict]:
        """Format all offers as accepted when LLM filtering fails in hybrid mode"""
        # Always use ALL initial offers for frontend display
        all_offers = self.initial_offers
        formatted_offers = []
        for offer in all_offers:
            formatted_offer = offer.copy()
            formatted_offer['accepted'] = True
            formatted_offer['rejection_reason'] = None
            formatted_offer['llm_filtered'] = False
            formatted_offer['fallback_reason'] = "LLM filtering unavailable - showing all offers"
            formatted_offers.append(formatted_offer)
        return formatted_offers

    def _format_offers_as_rejected(self, offers: List[Dict], llm_analysis: dict) -> List[Dict]:
        """Format all offers as rejected when LLM filtering results in 0 offers"""
        # Always use ALL initial offers for frontend display
        all_offers = self.initial_offers
        formatted_offers = []
        
        # Extract indicators for generating rejection reasons
        sentiment = llm_analysis.get('sentiment', 'neutral')
        emotion = llm_analysis.get('emotion', 'neutral')
        risk_patterns = llm_analysis.get('risk_patterns', [])
        offer_indicators = llm_analysis.get('offer_indicators', {})
        
        for offer in all_offers:
            formatted_offer = offer.copy()
            formatted_offer['accepted'] = False
            formatted_offer['llm_filtered'] = False
            
            # Generate specific rejection reason based on analysis
            rejection_reason = self._generate_enhanced_llm_rejection_reason(
                offer, sentiment, emotion, risk_patterns, offer_indicators
            )
            formatted_offer['rejection_reason'] = rejection_reason
            
            formatted_offers.append(formatted_offer)
        
        print(f"🔄 Formatted {len(formatted_offers)} offers as rejected due to filtering criteria")
        return formatted_offers

    def _generate_enhanced_llm_rejection_reason(self, offer: Dict, sentiment: str, emotion: str, risk_patterns: List[str], offer_indicators: dict) -> str:
        """Generate specific rejection reason for LLM-filtered offers with detailed reasoning"""
        reasons = []
        
        # Extract service usage and removal indicators
        service_usage = offer_indicators.get('service_usage', {})
        service_removal_interest = offer_indicators.get('service_removal_interest', {})
        budget_concern = offer_indicators.get('budget_concern_level', 'none')
        value_preference = offer_indicators.get('value_preference', 'balanced')
        
        # Check service removal interest FIRST (highest priority)
        offer_types = offer.get('product_types', [])
        
        if 'TV' in offer_types and service_removal_interest.get('tv_removal', False):
            reasons.append("Customer wants to remove TV service (tv_removal: true)")
        elif 'Mobile' in offer_types and service_removal_interest.get('mobile_removal', False):
            reasons.append("Customer wants to remove Mobile service (mobile_removal: true)")
        elif 'Internet' in offer_types and service_removal_interest.get('internet_removal', False):
            reasons.append("Customer wants to remove Internet service (internet_removal: true)")
        
        # Check service usage patterns (low usage)
        if 'TV' in offer_types:
            tv_usage = service_usage.get('tv_usage', 'unknown')
            if tv_usage == 'low':
                reasons.append(f"Customer barely watches TV (tv_usage: {tv_usage})")
        
        if 'Mobile' in offer_types:
            mobile_usage = service_usage.get('mobile_usage', 'unknown')
            if mobile_usage == 'low':
                reasons.append(f"Customer has low mobile usage (mobile_usage: {mobile_usage})")
                
        if 'Internet' in offer_types:
            internet_usage = service_usage.get('internet_usage', 'unknown')
            if internet_usage == 'low':
                reasons.append(f"Customer has low internet usage (internet_usage: {internet_usage})")
        
        # Check budget/price sensitivity
        if budget_concern == 'high' and offer['price_delta'] > 0:
            reasons.append(f"High budget concern with price increase (budget_concern_level: {budget_concern}, price_delta: ${offer['price_delta']})")
        elif budget_concern == 'medium' and offer['price_delta'] > 50:
            reasons.append(f"Medium budget concern with high price increase (budget_concern_level: {budget_concern}, price_delta: ${offer['price_delta']})")
        
        # Check billing complaints with price increases
        if sentiment in ['negative', 'very_negative'] and 'billing_complaint' in risk_patterns:
            if offer['price_delta'] > 0:
                reasons.append(f"Price increase conflicts with billing complaints (sentiment: {sentiment}, billing_complaint detected)")
            else:
                reasons.append(f"Not aligned with budget concerns despite discount (sentiment: {sentiment}, billing_complaint detected)")
        
        # Check emotional state with premium offers
        if emotion in ['anger', 'frustration'] and offer['category'] in ['upgrade', 'premium']:
            reasons.append(f"Customer {emotion} - premium offers not suitable (emotion: {emotion}, category: {offer['category']})")
        
        # Check competitor mentions with high-priced offers
        if 'competitor_mention' in risk_patterns and offer['price_delta'] > 50:
            reasons.append(f"High-priced offer when competitor mentioned (competitor_mention detected, price_delta: ${offer['price_delta']})")
        
        # Check service frustration with add-ons
        if 'service_frustration' in risk_patterns and 'add-on' in offer['category']:
            reasons.append(f"Customer frustrated with service - additional features not recommended (service_frustration detected)")
        
        # Check value preference
        if value_preference == 'price_focused' and offer['price_delta'] > 0:
            reasons.append(f"Customer is price-focused but offer increases cost (value_preference: {value_preference}, price_delta: ${offer['price_delta']})")
        
        # Default reason if no specific reason found
        if not reasons:
            if sentiment in ['negative', 'very_negative']:
                reasons.append(f"Not suitable for current customer sentiment (sentiment: {sentiment})")
            elif emotion in ['anger', 'frustration', 'sadness']:
                reasons.append(f"Not appropriate given customer emotion (emotion: {emotion})")
            else:
                reasons.append("Lower priority based on conversation analysis")
        
        return "; ".join(reasons)

def simulate_conversation_example():
    """Simulate the provided conversation example"""
    
    scorer = SimpleChurnScorer()
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

    # conversation = [
    #     ("Agent", "Thank you for calling customer Service, this is Jason speaking. May I have your account number please?"),
    #     ("Customer", "Sure, it's 29871003."),
    #     ("Agent", "Thanks, Mark. I'll need to verify your identity. I've just sent a 4-digit verification code to your registered mobile number ending in 6024. Could you read that out for me?"),
    #     ("Customer", "Yep, it's 9384."),
    #     ("Agent", "Perfect. Give me a moment while I pull up your account details... this may take 30 seconds. Do you mind holding?"),
    #     ("Customer", "That's fine."),
    #     ("Agent", "Thanks for waiting. I've got your account up. How can I help you today?"),
    #     ("Customer", "I just opened my bill and it's $200 again. Last month it was $180. Why is it going up every time?"),
    #     ("Agent", "I understand that's frustrating. Let me walk through your bill to identify the changes."),
    #     ("Customer", "I'm telling you, it's too high for what I'm using. I barely watch TV, and I don't even know what these charges are for."),
    #     ("Agent", "I see here that your promotional discount expired a month ago, and there's a rental charge for a second set-top box."),
    #     ("Customer", "No one told me the promo would end. Why wouldn't you notify me? This is not okay."),
    #     ("Agent", "You're absolutely right, and I apologize. We should have communicated that better. We might be able to lower your bill and give you better speeds or extras where it actually matters. Just a sec while I check what we can offer. Would you mind holding again for 30 seconds?"),
    #     ("Customer", "Sure."),
    #     ("Agent", "Thanks for holding. I've checked and we can offer you a plan that provides faster internet while keeping your TV, mobile and cybersecurity products at 185 dollar a month"),
    #     ("Customer", "Hmm. That is Still more than what I was paying. And honestly, TMobile are cheaper."),
    #     ("Agent", "I understand. Since you mentioned you barely watch TV, Do you want to remove that?"),
    #     ("Customer", "How much would that cost me?"),
    #     ("Agent", "So your new bill would be around $150/month & I can also add a 10 dollar discount if you subscribe for ebill which would make it $140 going forward. Your total savings would be 60 dollars a month"),
    #     ("Customer", "Okay, that sounds better. I appreciate you trying. But still, it feels like I have to call every few months just to keep the price reasonable. Just to confirm, I would still be keeping all the mobile lines right?"),
    #     ("Agent", "That's fair feedback, Sarah. I'll note that on your account. You shouldn't have to go through that and yes, you would keep all your products except TV"),
    #     ("Customer", "Alright, let's go with that then. Please make sure the new charges reflect next month."),
    #     ("Agent", "I've updated your plan and removed the rental. You'll see the changes in your next cycle. Anything else I can help you with today?"),
    #     ("Customer", "No, that's all. Thanks for the help, Jason."),
    #     ("Agent", "You're welcome. Thanks for being with HorizonConnect. Have a great day!")
    # ]
#     conversation = [
#     ("Agent", "Thank you for calling customer care, this is Rachel. May I have your account number please?"),
#     ("Customer", "Yes, it's 45120987."),
#     ("Agent", "Thanks, David. For security, I've just sent a 4-digit verification code to your mobile ending in 4431. Could you share that code?"),
#     ("Customer", "It's 7261."),
#     ("Agent", "Great, verified. One moment while I pull up your account… do you mind holding for 20 seconds?"),
#     ("Customer", "No problem."),
#     ("Agent", "Thanks for holding. I have your account open now. What can I help you with today?"),
#     ("Customer", "I just checked my bill—it's $220 this month. Last month it was $190. Why does it keep going up?"),
#     ("Agent", "I hear your concern. Let me review the line items on your bill."),
#     ("Customer", "Honestly, it feels like I'm paying for things I don't even use. I only use internet, and I barely touch the landline service."),
#     ("Agent", "I see that a promotional credit on your internet expired, and you've also got a long-distance package on the landline that's adding to the total."),
#     ("Customer", "No one ever explained that to me. It's really frustrating."),
#     ("Agent", "You're right, David, we should have given you a heads-up before the promo ended. Let me check if we can adjust your plan to lower your monthly cost. Would you mind holding for half a minute?"),
#     ("Customer", "Sure, go ahead."),
#     ("Agent", "Thank you. Here's what I found: We can move you to a package with higher internet speed and keep your mobile, while dropping the landline package you don't use. That would be $185/month."),
#     ("Customer", "Hmm. That's still higher than I'd like. I know Spectrum offers internet-only for $160."),
#     ("Agent", "I completely understand. If you're not using the landline, removing it will definitely help. On top of that, if you sign up for autopay, we can take another $15 off, bringing you to $170/month."),
#     ("Customer", "Okay, that's better. And I'd still keep my mobile line at the same price, right?"),
#     ("Agent", "Yes, your mobile stays as is, and the landline package would be removed. Your total monthly savings would be $50 compared to what you're paying now."),
#     ("Customer", "Alright, let's do that then. Please make sure the changes start next cycle."),
#     ("Agent", "I've updated your account and removed the long-distance package. You'll see the new pricing reflected on your next bill."),
#     ("Customer", "Perfect. That's all I needed. Thanks, Rachel."),
#     ("Agent", "You're very welcome, David. Thanks for being with HorizonConnect, and have a wonderful day!")
# ]

    
#     conversation = [
#     ("Agent", "Thank you for calling SkyConnect support, this is Rachel speaking. May I have your mobile number to pull up your account?"),
#     ("Customer", "Yeah, it's 917-445-2290."),
#     ("Agent", "Thank you, Alex. For security, I'll need to verify your date of birth, please."),
#     ("Customer", "It's January 12, 1990."),
#     ("Agent", "Perfect, thank you. I've got your account up. How can I help you today?"),
#     ("Customer", "My mobile data hasn't been working for three days. I can only use Wi-Fi. This is ridiculous—I pay for unlimited data."),
#     ("Agent", "I understand that's frustrating. Let's run a quick check together. Could you please turn off your phone's Wi-Fi and try accessing a website using only data?"),
#     ("Customer", "I've tried that ten times already—it doesn't load anything."),
#     ("Agent", "Thanks for confirming. I'm checking your network status... I see there's a service disruption in your area due to a tower upgrade. It's been ongoing since Sunday."),
#     ("Customer", "So you're telling me I've had no data for three days, and no one told me? That's unacceptable."),
#     ("Agent", "I completely understand your frustration, Alex. Unfortunately, the upgrade is still in progress. The estimated resolution is within 48 hours."),
#     ("Customer", "48 hours more? That's almost a week! I rely on data for work—what am I supposed to do?"),
#     ("Agent", "I'm truly sorry. What I can do is apply a temporary $25 credit to your account for the inconvenience. But the service will only be fully restored once the upgrade is complete."),
#     ("Customer", "Credit is fine, but honestly, this keeps happening. You guys need to fix your network."),
#     ("Agent", "I'll make sure your feedback is passed on to our technical team. I've logged this issue and added your account to the priority list."),
#     ("Customer", "Alright, but I'm not happy. Just confirm I'll see that credit on my next bill."),
#     ("Agent", "Yes, the $25 credit will reflect on your next billing cycle. I wish I had a faster fix for you today, Alex."),
#     ("Customer", "Okay… thanks, I guess."),
#     ("Agent", "Thank you for your patience. We appreciate you being with SkyConnect. Have a good day, and we'll keep working to get this resolved.")
# ]

    print("=== CHURN SCORING SIMULATION ===\n")
    print("Using VADER sentiment analysis with embedding-based pattern matching")
    print("")
    
    last_agent_message = ""
    
    for i, (speaker, text) in enumerate(conversation):
        print(f"\nTurn {i+1}")
        
        if speaker == "Customer":
            # Process customer message with previous agent context
            print(f"Customer: {text}")
            event = scorer.process_customer_message(text, last_agent_message)
            
            # Show current score after customer turns
            print(f"CURRENT SCORE: {scorer.get_current_score()}/100")
            
            # Get dynamic offer recommendations if churn score changed significantly
            if scorer.should_trigger_offer_update():
                offers = scorer.get_offers_for_agent(text)
                if offers:
                    print(f"\n🎯 DYNAMIC OFFERS (triggered by score change):")
                    for j, offer in enumerate(offers):
                        retention_flag = " [RETENTION]" if offer['retention_offer'] else ""
                        price_indicator = f" (${offer['price_delta']:+.0f})" if offer['price_delta'] != 0 else " (No cost)"
                        print(f"  {j+1}. {offer['title']}{retention_flag}")
                        print(f"     {offer['description']}")
                        print(f"     Value: {offer['value']} | Relevance: {offer['relevance']}%{price_indicator}")
            
        else:  # Agent
            # Store agent message for context but don't process for churn
            last_agent_message = text
            print(f"Agent: {text}")
    
    # Final assessment
    print(f"\n" + "="*60)
    print("FINAL CONVERSATION SUMMARY")
    print("="*60)
    print(f"Final Churn Score: {scorer.get_current_score()}/100")
    print(f"Total Customer Messages Processed: {len(scorer.risk_events)}")
    print(f"Total Conversation Turns: {len(conversation)}")
    
    return scorer

if __name__ == "__main__":
    # Run the simulation
    churn_scorer = simulate_conversation_example() 