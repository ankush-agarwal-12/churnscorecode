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
            'surprise': 5.0     # Low risk - unexpected situations
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
        
        # Analyze sentiment with agent context
        sentiment_score, sentiment_confidence = self.analyze_sentiment(customer_text, agent_context)
        print(f"Sentiment: score={sentiment_score:.3f}, confidence={sentiment_confidence:.3f}")
        
        # Analyze emotion
        emotion_result = self.analyze_emotion(customer_text)
        print(f"Emotion: {emotion_result['dominant_emotion']} ({emotion_result['dominant_score']:.3f})")
        
        # Detect risk patterns and update context
        pattern_risk, detected_patterns = self.detect_risk_patterns_embedding(customer_text, agent_context)
        self.update_conversation_context(customer_text, detected_patterns)
        
        # Calculate sentiment risk
        sentiment_risk = 0.0
        if sentiment_score < -0.6:  # Very negative
            sentiment_risk = 30.0
        elif sentiment_score < -0.3:  # Moderately negative
            sentiment_risk = 20.0
        elif sentiment_score < -0.1:  # Slightly negative
            sentiment_risk = 10.0
        elif sentiment_score > 0.35:  # Positive
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
        
        # Update cumulative score with detailed calculation
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
        self.conversation_history.append({
            'timestamp': timestamp,
            'speaker': 'Customer',
            'text': customer_text,
            'sentiment': sentiment_score
        })
        
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

    def reset_conversation(self):
        """Reset for a new conversation"""
        self.current_score = self.baseline_score
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

def simulate_conversation_example():
    """Simulate the provided conversation example"""
    
    scorer = SimpleChurnScorer()
    
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
#     conversation = [
#     ("Agent", "Thank you for calling customer care, this is Rachel. May I have your account number please?"),
#     ("Customer", "Yes, it’s 45120987."),
#     ("Agent", "Thanks, David. For security, I’ve just sent a 4-digit verification code to your mobile ending in 4431. Could you share that code?"),
#     ("Customer", "It’s 7261."),
#     ("Agent", "Great, verified. One moment while I pull up your account… do you mind holding for 20 seconds?"),
#     ("Customer", "No problem."),
#     ("Agent", "Thanks for holding. I have your account open now. What can I help you with today?"),
#     ("Customer", "I just checked my bill—it’s $220 this month. Last month it was $190. Why does it keep going up?"),
#     ("Agent", "I hear your concern. Let me review the line items on your bill."),
#     ("Customer", "Honestly, it feels like I’m paying for things I don’t even use. I only use internet, and I barely touch the landline service."),
#     ("Agent", "I see that a promotional credit on your internet expired, and you’ve also got a long-distance package on the landline that’s adding to the total."),
#     ("Customer", "No one ever explained that to me. It’s really frustrating."),
#     ("Agent", "You’re right, David, we should have given you a heads-up before the promo ended. Let me check if we can adjust your plan to lower your monthly cost. Would you mind holding for half a minute?"),
#     ("Customer", "Sure, go ahead."),
#     ("Agent", "Thank you. Here’s what I found: We can move you to a package with higher internet speed and keep your mobile, while dropping the landline package you don’t use. That would be $185/month."),
#     ("Customer", "Hmm. That’s still higher than I’d like. I know Spectrum offers internet-only for $160."),
#     ("Agent", "I completely understand. If you’re not using the landline, removing it will definitely help. On top of that, if you sign up for autopay, we can take another $15 off, bringing you to $170/month."),
#     ("Customer", "Okay, that’s better. And I’d still keep my mobile line at the same price, right?"),
#     ("Agent", "Yes, your mobile stays as is, and the landline package would be removed. Your total monthly savings would be $50 compared to what you’re paying now."),
#     ("Customer", "Alright, let’s do that then. Please make sure the changes start next cycle."),
#     ("Agent", "I’ve updated your account and removed the long-distance package. You’ll see the new pricing reflected on your next bill."),
#     ("Customer", "Perfect. That’s all I needed. Thanks, Rachel."),
#     ("Agent", "You’re very welcome, David. Thanks for being with HorizonConnect, and have a wonderful day!")
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